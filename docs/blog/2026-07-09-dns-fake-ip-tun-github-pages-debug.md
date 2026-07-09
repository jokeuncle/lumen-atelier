---
title: "DNS、fake-ip 与 TUN：一次 GitHub Pages 无法访问的排障复盘"
date: 2026-07-09
source: "Lumen Atelier network debugging note"
---

# DNS、fake-ip 与 TUN：一次 GitHub Pages 无法访问的排障复盘

这次的问题很典型：

```text
手机能打开 jokeuncle.github.io
同一时间，这台电脑打不开
```

第一反应很容易是：

```text
是不是 DNS 被污染了？
```

最后的结论不是这么简单。更准确地说：

**这台电脑上的 Clash Verge / mihomo 开了 TUN 和 fake-ip，普通 DNS 查询被本地代理接管，`jokeuncle.github.io` 被映射到 `198.18.0.100`。真正导致打不开的不是这个 fake-ip 本身，而是当前选中的代理节点到 GitHub Pages / GitHub 静态资源这一类域名的链路不通。切到 `Singapore 01` 后访问恢复。**

这篇不是记录“哪个节点能用”。节点会变，规则会变，网络也会变。真正值得学的是背后的排障模型：

```text
域名解析层 -> 本机代理层 -> 路由/TUN 层 -> TCP 层 -> TLS/HTTP 层 -> 代理节点链路
```

所谓“一眼精通”，其实不是玄学，而是每一层都问一个很具体的问题。

## 1. 先把现象拆成层：打不开到底卡在哪里

“打不开网页”太粗了。它可能至少有六种原因：

```text
1. DNS 查不到域名
2. DNS 查到了错误 IP
3. TCP 连不上 443 端口
4. TCP 连上了，但 TLS 握手失败
5. TLS 成功了，但 HTTP 返回 403/404/5xx
6. 页面 HTML 能拿到，但 JS/CSS/图片资源失败
```

所以第一步不是换 DNS，也不是清缓存，而是把问题问细：

```text
这台电脑到底拿到了什么 IP？
它是直连还是走代理？
TCP 有没有连上？
TLS 有没有拿到服务器证书？
HTTP 有没有状态码？
```

这次最关键的三个事实是：

```text
本机普通 DNS: jokeuncle.github.io -> 198.18.0.100
DoH 结果:      jokeuncle.github.io -> 185.199.108/109/110/111.153
curl 现象:     TCP/CONNECT 成功，但 TLS ClientHello 后失败
```

这三条合在一起，就已经把问题从“也许是 DNS”推进到了“本地代理/TUN 和代理节点链路”。

## 2. `198.18.0.100` 不一定是 DNS 污染，它更像 fake-ip

本机查到：

```text
jokeuncle.github.io.  1  IN  A  198.18.0.100
```

如果只看这一行，容易紧张：GitHub Pages 不应该是这个 IP。

但 `198.18.0.0/15` 这段地址经常被代理软件拿来做 fake-ip。它的作用不是让你真的去访问一个公网服务器，而是做一个“域名到假 IP”的本地映射。

人话版：

```text
浏览器问：jokeuncle.github.io 是谁？
本地代理答：先记成 198.18.0.100。
浏览器连 198.18.0.100。
TUN 接住这条连接，再查回它原本对应的域名 jokeuncle.github.io。
代理按域名规则决定走哪个节点。
```

所以 fake-ip 的关键不是“这个 IP 是真的 GitHub 吗”。它本来就不是真的。关键是：

```text
本机有没有一个 TUN/代理系统负责把这个 fake-ip 接住？
接住以后，它有没有把流量送到正确的出口？
```

这次路由表给出了答案：

```text
route to: 198.18.0.100
gateway: 198.18.0.1
interface: utun1024
```

`utun1024` 是 macOS 上的虚拟网络接口。`198.18.0.1` 是本地 TUN 网关。再结合进程里看到 Clash Verge / mihomo，就能判断：

```text
198.18.0.100 是 Clash fake-ip 体系里的地址。
它不是独立的运营商 DNS 污染证据。
```

更容易踩坑的一点是：即使你写了 `dig @1.1.1.1 jokeuncle.github.io`，也不一定真的绕开了本地代理。

因为 TUN 配置里可以有：

```text
dns-hijack: any:53
```

意思是所有普通 53 端口 DNS 请求都会被接管。你以为自己问的是 `1.1.1.1`，实际这条 UDP/TCP 53 流量已经被本地代理截走了。

这也是为什么这次 `@1.1.1.1`、`@8.8.8.8`、`@223.5.5.5` 都返回了同一个 fake-ip。它们不是三个公共 DNS 都被污染，而是本机普通 DNS 路径已经被代理统一接管。

## 3. DoH 的价值：拿一个“公网视角”的对照组

为了对照，可以用 DNS over HTTPS：

```text
https://cloudflare-dns.com/dns-query?name=jokeuncle.github.io&type=A
https://dns.google/resolve?name=jokeuncle.github.io&type=A
```

这次 DoH 返回：

```text
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```

这说明从公网 DNS 视角看，GitHub Pages 的地址是正常的。

但 DoH 只能回答一个问题：

```text
如果绕过本机普通 DNS 劫持，公网 DNS 会返回什么？
```

它不能单独证明网页一定能打开。网页能不能打开，还要看后面的 TCP、TLS、代理出口和节点链路。

这就是排障时很重要的一条原则：

```text
DNS 正常，不代表访问正常。
DNS 异常，也不代表根因一定是 DNS。
```

DNS 只是第一层。

## 4. `curl -Iv` 的价值：看 TCP、TLS、HTTP 分别走到哪一步

这次 `curl -Iv https://jokeuncle.github.io/` 的关键输出不是最后那句报错，而是报错前发生了什么：

```text
Uses proxy env variable HTTPS_PROXY == http://127.0.0.1:7897
Connected to 127.0.0.1 port 7897
CONNECT tunnel established, response 200
TLS handshake, Client hello
LibreSSL SSL_connect: SSL_ERROR_SYSCALL
```

逐层翻译：

```text
HTTPS_PROXY 生效了，curl 先连本机 7897 端口。
本机代理接受了 CONNECT 请求，并返回 200。
也就是说 HTTP 代理入口没坏。
然后 curl 发出了 TLS ClientHello。
但没有顺利收到服务端 TLS ServerHello。
```

这就把问题缩小了：

```text
不是 curl 没走代理。
不是本机代理端口没开。
不是 TCP 到本机代理失败。
问题发生在代理把这条 TLS 流量送往目标站的后半段。
```

更关键的是，同一个代理下访问 `github.com` 是成功的：

```text
github.com -> HTTP/2 200
```

但访问这些域名失败：

```text
jokeuncle.github.io
pages.github.com
raw.githubusercontent.com
githubusercontent.com
```

这说明问题不是“GitHub 全挂”，而更像是：

```text
当前节点对 GitHub Pages / GitHub 静态资源 CDN 这一组目标链路不通。
```

这是一种很常见的代理排障结论：同一个大站下面，不同域名、不同 CDN、不同 IP 段，可能走到完全不同的边缘网络。`github.com` 能打开，不代表 `github.io`、`raw.githubusercontent.com`、`githubusercontent.com` 也一定能打开。

## 5. `--noproxy` 的陷阱：绕过环境变量，不等于绕过 TUN

排障时我也用了：

```bash
curl --noproxy '*' -Iv https://jokeuncle.github.io/
```

这个参数只表示：

```text
不要使用 HTTP_PROXY / HTTPS_PROXY 这些环境变量。
```

它不表示：

```text
绕过系统路由。
绕过 TUN。
绕过 Clash 的 fake-ip。
```

所以在开 TUN 的机器上，即使 `curl --noproxy '*'`，连接仍然可能走：

```text
应用 -> 系统网络栈 -> utun 虚拟网卡 -> Clash/mihomo -> 代理节点
```

这次就是这样。`--noproxy` 后，curl 直接连的是 fake-ip：

```text
Host jokeuncle.github.io resolved to 198.18.0.100
Trying 198.18.0.100:443
```

但路由表显示这条连接仍然进入 `utun1024`。所以它不是“真正直连公网 GitHub Pages”，只是绕过了显式 HTTP 代理环境变量。

这条经验很重要：

```text
HTTP 代理和 TUN 是两套入口。
关掉 curl 的 proxy 参数，不等于关掉系统级透明代理。
```

## 6. 规则没错时，继续看“规则选中的节点”

Clash 配置里有规则：

```text
DOMAIN-SUFFIX,github.io,Proxies
DOMAIN-SUFFIX,github.com,Proxies
DOMAIN-SUFFIX,githubusercontent.com,Proxies
```

所以 `github.io` 没有被错误地放到 `DIRECT`。它确实进入了 `Proxies` 代理组。

下一步就不是纠结 DNS，而是看：

```text
Proxies 当前选中了谁？
这个节点到目标站是否真的可用？
```

当时 `Proxies` 选中的是：

```text
Taiwan 01
```

Clash 自己的 delay API 结果也很直接：

```text
Taiwan 01 -> https://github.com/              OK
Taiwan 01 -> https://jokeuncle.github.io/     503
```

这条证据非常有力。它说明：

```text
同一个节点不是完全坏。
但它到 GitHub Pages 这条目标链路坏。
```

换几个候选节点测试后，下面这些能访问 GitHub Pages：

```text
Hong Kong 01
Japan 01
Japan 02
Singapore 01
```

最终切到：

```text
Singapore 01
```

再次访问：

```text
https://jokeuncle.github.io/ -> HTTP 200
```

这就是最终闭环：

```text
不是网站没发布。
不是浏览器缓存。
不是 hosts。
不是单纯 DNS 污染。
是当前代理节点到 GitHub Pages / 静态资源域名不通。
```

## 7. 为什么手机能打开，电脑不能打开

这个问题现在也能解释了。

手机和电脑表面上访问同一个域名，但真实路径可能完全不同：

```text
手机:
浏览器 -> 手机网络/DNS/VPN/代理 -> GitHub Pages

电脑:
浏览器 -> macOS 网络栈 -> Clash TUN/fake-ip -> 当前 Proxies 节点 -> GitHub Pages
```

只要其中一层不同，结果就可能不同。

所以“手机能打开”只能说明：

```text
网站本身大概率正常。
公网 GitHub Pages 大概率正常。
```

它不能说明：

```text
电脑的 DNS 正常。
电脑的代理规则正常。
电脑当前代理节点到该 CDN 正常。
```

这也是排障里的基本思路：

```text
两台设备结果不同，不要只比较域名。
要比较完整路径。
```

## 8. 这次排障用到的命令，分别在验证什么

下面不是让你死记命令，而是记住每个命令回答的问题。

### DNS 层：本机拿到了什么地址

```bash
dig jokeuncle.github.io A
dscacheutil -q host -a name jokeuncle.github.io
```

回答：

```text
应用在本机 DNS 体系里会看到什么 IP？
```

这次答案是 fake-ip：

```text
198.18.0.100
```

### 公网 DNS 对照：绕开普通 53 端口 DNS

```bash
curl 'https://dns.google/resolve?name=jokeuncle.github.io&type=A'
```

回答：

```text
公网 DNS 视角下，这个域名现在应该解析到哪里？
```

这次答案是 GitHub Pages 的 `185.199.*.153`。

### 代理入口：curl 是否在用本机代理

```bash
env | rg -i '^(http|https|all|no)_proxy='
curl -Iv https://jokeuncle.github.io/
```

回答：

```text
命令行请求是否被 HTTP_PROXY / HTTPS_PROXY 接管？
本机代理端口是否接受 CONNECT？
```

这次答案是：

```text
HTTPS_PROXY=http://127.0.0.1:7897
CONNECT 200
```

### 路由层：fake-ip 被谁接住

```bash
route -n get 198.18.0.100
```

回答：

```text
访问 fake-ip 时，系统会把包发到哪个接口？
```

这次答案是：

```text
utun1024
```

### 进程层：谁在开 TUN 和代理端口

```bash
ps aux | rg -i 'mihomo|clash|sing-box|surge'
```

回答：

```text
本机到底是哪套代理软件在接管网络？
```

这次答案是 Clash Verge / mihomo。

### TLS/HTTP 层：失败发生在握手前后

```bash
curl -Iv https://jokeuncle.github.io/
```

回答：

```text
TCP 连了吗？
TLS 证书拿到了吗？
HTTP 状态码出来了吗？
```

这次在坏节点上停在 TLS ClientHello 后；切换节点后拿到 `HTTP/2 200`。

## 9. 一张可复用的判断表

以后遇到“手机能上，电脑不能上”，可以按这张表看。

| 现象 | 更像什么问题 | 下一步 |
|---|---|---|
| DNS 无结果 | DNS 服务或域名配置问题 | 换 DoH / 查权威解析 / 查域名状态 |
| DNS 返回 `198.18.*` | 本机代理 fake-ip | 查 TUN、路由、代理规则 |
| `curl` 显示 `CONNECT 200` 后 TLS 失败 | 代理出口或目标链路问题 | 换节点 / 查该域名规则 / 对比同站不同域名 |
| `github.com` 能开，`github.io` 不能开 | GitHub 不同域名/CDN 路径差异 | 单独测 `github.io`、`raw.githubusercontent.com`、`githubusercontent.com` |
| `--noproxy` 后仍走 `utun` | TUN 透明代理仍生效 | 关 TUN 或改系统路由测试 |
| TLS 证书 issuer 变成本地安全软件 | HTTPS 被本机安全软件检查 | 检查安全软件 HTTPS scanning 设置 |
| 换节点后立刻恢复 | 节点链路问题 | 固定可用节点或为该域名建更稳定规则 |

这张表比“换 DNS 试试”更有用，因为它把症状和层次对应起来。

## 10. 最后总结：这次到底学会了什么

这次的核心不是某个命令，而是五个判断：

```text
1. 看到 198.18.*，先想到 fake-ip，不要立刻判定 DNS 污染。
2. `dig @1.1.1.1` 在 TUN + dns-hijack 下也可能被本机接管。
3. DoH 可以作为公网 DNS 对照组，但不能证明访问链路一定通。
4. `CONNECT 200` 只说明本机 HTTP 代理入口正常，不说明远端链路正常。
5. `github.com` 正常不代表 `github.io`、`raw.githubusercontent.com`、`githubusercontent.com` 正常。
```

如果要把这套方法压成一句话：

**不要问“是不是 DNS 污染”，要问“请求现在卡在哪一层”。**

当你能把“打不开”拆成 DNS、代理、路由、TCP、TLS、HTTP、节点链路这几层时，看起来就会像“一眼看懂”。其实只是每一步都有明确证据。

## 自检题

1. 为什么 `198.18.0.100` 更像 Clash fake-ip，而不是 GitHub Pages 的真实 IP？
2. 为什么 `curl --noproxy '*'` 不能证明已经绕过了 TUN？
3. 如果 `github.com` 能打开，但 `raw.githubusercontent.com` 不能打开，你会优先查 DNS、规则，还是节点链路？为什么？
4. `CONNECT tunnel established, response 200` 说明了什么？没有说明什么？


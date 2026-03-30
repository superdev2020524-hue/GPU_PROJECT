# XCP-ng host (server2): HTTPS and orchestration recovery

**Host:** management IP `10.25.33.20` (dom0 `xcp-ng-sfgagrpq`), XCP-ng 8.3.

## What went wrong

After a round of package updates, the web UI and Xen Orchestra both failed against this host: browsers showed connection resets on `https://…`, and the management API never came up cleanly. On the server itself, nothing was listening on port 80, and the `xapi_startup.cookie` file under `/var/run` never appeared—classic signs that the Xen API service had not finished starting.

The logs pointed to database sync during startup, with errors mentioning an unknown network RPC method. Comparing installed RPMs showed the real issue: **xapi-core** had been updated to the **26.1.x** line, but **xcp-networkd**, **xapi-xe**, and **xapi-rrd2csv** were still on **25.6.x**. Those components talk to each other over a small RPC protocol; when one side is a full release ahead of the other, startup fails and the HTTP stack never binds—hence stunnel on 443 had nothing to talk to on the inside.

A second problem showed up when we tried to fix it with `yum`: downloads failed because the host could not resolve `mirrors.xcp-ng.org`. The only DNS server in `/etc/resolv.conf` was an internal address that wasn’t answering for public names. Internet reachability was fine (ping to public IPs worked), so we added a public resolver (`8.8.8.8`) and kept a backup of the old file before changing it.

## What I did

1. Align the toolstack: `yum update` for **xcp-networkd**, **xapi-xe**, and **xapi-rrd2csv** to **26.1.3-1.3** (same family as **xapi-core**).
2. Run **`xe-toolstack-restart`** on the dom0 (this restarts the management stack on the server—not a reboot of your PC).
3. Confirm: port **80** in use by **xapi**, **443** still fronted by **stunnel**, cookie file present, `curl` to `http://127.0.0.1:80` returning a normal redirect.

After that, **https://10.25.33.20** loads the bundled **XO Lite** UI again. Login there is **root** with the **same password** as SSH; if the UI says the password is wrong, fix or verify `root` on the host with `passwd` over SSH—there is no separate “XO password” for this screen.

## Follow-up worth doing

- Fix DNS properly on the management network (or point the host at resolvers that can resolve both internal and XCP-ng mirror names) so future `yum` runs don’t depend on a one-off edit to `resolv.conf`.
- Keep future updates **complete** for the toolstack: letting `xapi-core` jump ahead without updating **xcp-networkd** in the same maintenance window will recreate this class of failure.

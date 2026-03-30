# Restart mediator on host

From your machine (where the gpu repo lives), run:

```bash
cd /home/david/Downloads/gpu/phase3
python3 connect_host.py "pkill -f mediator_phase3 2>/dev/null; sleep 2; cd /root/phase3 && nohup ./mediator_phase3 2>/tmp/mediator.log </dev/null & disown; sleep 2; pgrep -a mediator"
```

Or on the host itself (e.g. `ssh root@10.25.33.10`):

```bash
pkill -f mediator_phase3
sleep 2
cd /root/phase3 && nohup ./mediator_phase3 2>/tmp/mediator.log </dev/null & disown
sleep 2
pgrep -a mediator
```

Host is `root@10.25.33.10` (from `vm_config.py`: MEDIATOR_HOST, MEDIATOR_USER).

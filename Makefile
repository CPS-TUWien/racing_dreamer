.PHONY: download_pixi

download_pixi:
	@scp -r axel@pixi:~/Projects/acme_racing/logs/experiments ./logs/

download_frida:
	@scp -P 2206 -r axelbr@frida:~/Projects/acme_racing/logs/experiments ./logs/

download_t400:
	@scp -r axel@t400-208n5:~/Projects/acme_racing/logs/experiments ./logs/
.PHONY: download_pixie

download_pixi:
	@scp -r axel@128.130.39.92:~/Projects/acme_racing/logs/experiments ./logs/

download_frida:
	@scp -r -P 2206 axelbr@frida.hopto.org:~/Projects/acme_racing/logs/experiments ./logs/
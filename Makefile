.PHONY: download_pixie

download_pixi:
	@scp -r axel@128.130.39.92:~/Projects/acme_racing/logs/experiments ./logs/

download_frida:
	@scp -P 2206 -r axelbr@frida.hopto.org:~/Projects/acme_racing/logs/experiments ./logs/
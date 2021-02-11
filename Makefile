.PHONY: download_pixie

download_pixi:
	@scp -r axel@128.130.39.92:~/Projects/acme_racing/logs/experiments ./logs/

download_frida:
	@scp -P 2206 -r axelbr@frida.hopto.org:~/Projects/acme_racing/logs/experiments ./logs/

download_t400:
	@scp -r axel@t400-208n5:~/Projects/acme_racing/logs/experiments ./logs/
token=`cat .token`
psw=`cat .password`
logdir=${1}

echo -e "[Info] Creating tar ball for log dir ${logdir}\n"
tarball="$(who | cut -d ' ' -f 1)_$(date '+%d%m%Y_%H%M%S')_$(basename -- ${logdir}).tar"
tar cvf ${tarball} --exclude='*/videos' --exclude='*/checkpoints' ${logdir}
echo -e "[Info] Created ${tarball}\n"

echo "[Info] Uploading ${tarball} ..."
time curl -u ${token}:${psw} -T ${tarball} "https://owncloud.tuwien.ac.at/public.php/webdav/${tarball}"
echo "[Info] Done."

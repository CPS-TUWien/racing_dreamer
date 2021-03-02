token=`cat .token`
psw=`cat .password`
logdir=${1}

echo -e "[Info] Creating tar ball for log dir ${logdir}\n"
tarball="${USER}_$(date '+%d%m%Y_%H%M%S')_$(basename -- ${logdir}).tar"

# note: since tar v 1.29, exclude works only on successive arguments
# for this reason we exclude checkpoints/*pkl after storing the best ones
tar cvf ${tarball} --exclude='*/checkpoints' ${logdir}
echo -e "[Info] Created ${tarball}\n"

echo "[Info] Uploading ${tarball} ..."
time curl -u ${token}:${psw} -T ${tarball} "https://owncloud.tuwien.ac.at/public.php/webdav/${tarball}"
echo "[Info] Done."
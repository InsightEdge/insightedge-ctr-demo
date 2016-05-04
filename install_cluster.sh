export ARTIFACT_DOWNLOAD_COMMAND="wget -O gigaspaces-insightedge-0.4.0-SNAPSHOT.zip https://www.dropbox.com/s/4e78nzm9pu7h1kf/gigaspaces-insightedge-0.4.0-SNAPSHOT.zip?dl=1"

./sbin/insightedge.sh --mode remote-master --hosts 10.8.1.115 --user ec2-user --key ~/Soft/aws-gigaspaces/fe-shared.pem --install --path /home/ec2-user/ie-0.4 --master 10.8.1.115
./sbin/insightedge.sh --mode remote-slave --hosts 10.8.1.117,10.8.1.118,10.8.1.116,10.8.1.119 --user ec2-user  --key ~/Soft/aws-gigaspaces/fe-shared.pem --install --path /home/ec2-user/ie-0.4 --master 10.8.1.115 -s 8G

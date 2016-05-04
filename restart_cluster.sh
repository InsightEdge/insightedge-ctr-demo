export MASTER=10.8.1.115
export SLAVES=10.8.1.117,10.8.1.118,10.8.1.116,10.8.1.119
export IE_PATH=/home/ec2-user/ie-0.4

~/Soft/gigaspaces-insightedge-0.4.0-SNAPSHOT/sbin/insightedge.sh --mode undeploy \
  --master $MASTER --group dev-env

~/Soft/gigaspaces-insightedge-0.4.0-SNAPSHOT/sbin/insightedge.sh --mode remote-master --hosts $MASTER --user ec2-user --key ~/Soft/aws-gigaspaces/fe-shared.pem --path $IE_PATH --master $MASTER

~/Soft/gigaspaces-insightedge-0.4.0-SNAPSHOT/sbin/insightedge.sh --mode remote-slave --hosts $SLAVES \
  --user ec2-user --key ~/Soft/aws-gigaspaces/fe-shared.pem \
  --path $IE_PATH --master $MASTER -s 8G

~/Soft/gigaspaces-insightedge-0.4.0-SNAPSHOT/sbin/insightedge.sh --mode deploy \
  --master $MASTER --group dev-env

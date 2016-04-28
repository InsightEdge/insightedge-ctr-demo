sbt clean assembly
scp -i ~/Soft/aws-gigaspaces/fe-shared.pem ~/Projects/insightedge-ctr-demo/target/scala-2.10/insightedge-ctr-demo-assembly-1.0.0.jar ec2-user@10.8.1.115:/home/ec2-user/avazu_ctr
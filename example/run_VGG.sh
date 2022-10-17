echo "offset    time"
for i in {0..10}
do
    ../bin/VGG -o $[i*10]
done
# SETUP the ramfs
# Set up the SIZE carefully (in bytes)

Ramfs_size=1G
echo "SETUP RAMFS size: $Ramfs_size"
sudo mkdir -p /tmp2
sudo mount -t ramfs -o size=$Ramfs_size ramfs /tmp2
sudo chmod 777 /tmp2

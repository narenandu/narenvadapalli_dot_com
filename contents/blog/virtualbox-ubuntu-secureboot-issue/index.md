---
title: Secureboot + Ubuntu + VirtualBox Signing kernel modules
date: 2021-05-09
template: blog
image: ./image.png
banner: ./banner.jpg
description: Set of steps required for dealing with secureboot on Ubuntu where VirutalBox service has issues
---

### Motivation

```
minikube start
```

was throwing errors like the following

```
  minikube start
😄  minikube v1.18.1 on Ubuntu 21.04
✨  Automatically selected the virtualbox driver
👍  Starting control plane node minikube in cluster minikube
🔥  Creating virtualbox VM (CPUs=2, Memory=2900MB, Disk=20000MB) ...
🤦  StartHost failed, but will try again: creating host: create: precreate: We support Virtualbox starting with version 5. Your VirtualBox install is "WARNING: The vboxdrv kernel module is not loaded. Either there is no module\n         available for the current kernel (5.11.0-16-generic) or it failed to\n         load. Please recompile the kernel module and install it by\n\n           sudo /sbin/vboxconfig\n\n         You will not be able to start VMs until this problem is fixed.\n6.1.22r144080". Please upgrade at https://www.virtualbox.org
🔥  Creating virtualbox VM (CPUs=2, Memory=2900MB, Disk=20000MB) ...
😿  Failed to start virtualbox VM. Running "minikube delete" may fix it: creating host: create: precreate: We support Virtualbox starting with version 5. Your VirtualBox install is "WARNING: The vboxdrv kernel module is not loaded. Either there is no module\n         available for the current kernel (5.11.0-16-generic) or it failed to\n         load. Please recompile the kernel module and install it by\n\n           sudo /sbin/vboxconfig\n\n         You will not be able to start VMs until this problem is fixed.\n6.1.22r144080". Please upgrade at https://www.virtualbox.org
❗  Startup with virtualbox driver failed, trying with alternate driver ssh: Failed to start host: creating host: create: precreate: We support Virtualbox starting with version 5. Your VirtualBox install is "WARNING: The vboxdrv kernel module is not loaded. Either there is no module\n         available for the current kernel (5.11.0-16-generic) or it failed to\n         load. Please recompile the kernel module and install it by\n\n           sudo /sbin/vboxconfig\n\n         You will not be able to start VMs until this problem is fixed.\n6.1.22r144080". Please upgrade at https://www.virtualbox.org
💀  Removed all traces of the "minikube" cluster.

❌  Exiting due to MK_USAGE: No IP address provided. Try specifying --ssh-ip-address, or see https://minikube.sigs.k8s.io/docs/drivers/ssh/

```

### Resolution

Had to check multiple resources to make this a working solution. Quoting all the original resources

-   https://stegard.net/2016/10/virtualbox-secure-boot-ubuntu-fail/
-   http://askubuntu.com/questions/760671/could-not-load-vboxdrv-after-upgrade-to-ubuntu-16-04-and-i-want-to-keep-secur

#### Install the virtualbox manually

```
sudo apt-get update
sudo apt-get install virtualbox-6.1
```

#### Sign the modules for secureboot

```
sudo -i
mkdir /root/module-signing
cd /root/module-signing
openssl req -new -x509 -newkey rsa:2048 -keyout MOK.priv -outform DER -out MOK.der -nodes -days 36500 -subj "/CN=Descriptive common name/"

mokutil --import /root/module-signing/MOK.der
# Input a simple password
```

#### Restart the machine

-   During the boot when prompted choose `Enroll MOK`
-   You will see the keys that were created and signed and choose `Continue`
-   `Reboot`

#### Create a bash script to sign the kernel modules

```
sudo -i
touch /root/module-signing/sign-vbox-modules
vi /root/module-signing/sign-vbox-modules
```

Paste the following in to the script file (hit `i` to be in insert mode)

```
#!/bin/bash

for modfile in $(dirname $(modinfo -n vboxdrv))/*.ko; do
  echo "Signing $modfile"
  /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 \
                                /root/module-signing/MOK.priv \
                                /root/module-signing/MOK.der "$modfile"
done
```

Then hit `ESC + wq` to save and quit the file

Execute the script after updating the permissions

```
chmod 700 /root/module-signing/sign-vbox-modules
/root/module-signing/sign-vbox-modules
```

Sample output should look like the following

```
#  /root/module-signing/sign-vbox-modules
Signing /lib/modules/5.11.0-16-generic/updates/dkms/vboxdrv.ko
Signing /lib/modules/5.11.0-16-generic/updates/dkms/vboxnetadp.ko
Signing /lib/modules/5.11.0-16-generic/updates/dkms/vboxnetflt.ko
```

#### Start Virtualbox

```
modprobe vboxdrv
```

### Check

```
minikube start
```

should now work as expected and start the local cluster

```
$  minikube start
😄  minikube v1.18.1 on Ubuntu 21.04
✨  Automatically selected the virtualbox driver. Other choices: none, ssh
👍  Starting control plane node minikube in cluster minikube
🔥  Creating virtualbox VM (CPUs=2, Memory=2900MB, Disk=20000MB) ...
🐳  Preparing Kubernetes v1.20.2 on Docker 20.10.3 ...
    ▪ Generating certificates and keys ...
    ▪ Booting up control plane ...
    ▪ Configuring RBAC rules ...
🔎  Verifying Kubernetes components...
    ▪ Using image gcr.io/k8s-minikube/storage-provisioner:v4
🌟  Enabled addons: storage-provisioner, default-storageclass
🏄  Done! kubectl is now configured to use "minikube" cluster and "default" namespace by default

```

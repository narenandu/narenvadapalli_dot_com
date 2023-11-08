---
title: How to find a linux machine is a VM (Virtual Machine) or a Bare Metal
date: 2023-11-07
template: blog
image: ./image.jpg
banner: ./banner.jpg
description: If you can SSH into a linux machine and want to find out if its baremetal or Virtual Machine
---


## Motivation

You might be given access to a linux machine via SSH crendetials. But you are really curious, if this machine is actually a bare metal or a virtual machine

## Commands to help out
There are many different commands that gives us the required information, lets see...


### virt-what

`virt-what` command will give the information about any virtualization platform if relevant

#### Examples

**Virtual Machines**

VMWare VM
```
# virt-what
vmware
```

VirtualBox VM
```
# virt-what
virtualbox
kvm
```

**Baremetal**
```
# virt-what
#NOTHING - NO OUTPUT
```

### systemd-detect-virt

`systemd-detect-virt` command would give the virtualization platform if relevant. Also the exit status from this command can be used as the check

#### Examples

**Virtual Machines**

VMWare VM
```
# systemd-detect-virt
vmware

# echo $?
0
```

VirtualBox VM
```
# systemd-detect-virt
kvm

# echo $?
0
```

**Baremetal**
```
# systemd-detect-virt
none

# echo $?
1
```

### hostnamectl

`hostnamectl | grep Chassis` command would tell us if the machine is a VM or Baremetal

#### Examples

**Virtual Machines**

VMWare VM
```
# hostnamectl | grep Chassis
           Chassis: vm
```

VirtualBox VM
```
# hostnamectl | grep Chassis
           Chassis: vm
```

**Baremetal**
```
# hostnamectl | grep Chassis
           Chassis: desktop
```

### dmidecode -s system-manufacturer

`dmidecode -s system-manufacturer` command would give the information about the system manufacturer

#### Examples

**Virtual Machines**

VMWare VM
```
# dmidecode -s system-manufacturer
VMware, Inc.
```

VirtualBox VM
```
# dmidecode -s system-manufacturer
innotek GmbH
```

**Baremetal**
```
# dmidecode -s system-manufacturer
Dell Inc.
```

### dmesg
`dmesg | grep -i virtual` command would output the logs and there is typically indication about the Virtualization platform

#### Examples

**Virtual Machines**

VMWare VM
```
# dmesg | grep -i virtual
[    0.000000] DMI: VMware, Inc. VMware Virtual Platform/XXXX Desktop Reference Platform, BIOS X.00 MM/DD/YYYY
[    0.000000] Booting paravirtualized kernel on VMware hypervisor
[    0.613050] systemd[1]: Detected virtualization vmware.
[    0.634050] systemd[1]: Starting Setup Virtual Console...
[    0.734374] VMware vmxnet3 virtual NIC driver - version
.......
# Some more relevant messages
```

VirtualBox VM
```
# dmesg | grep -i virtual
[    0.000000] DMI: innotek GmbH VirtualBox/VirtualBox, BIOS VirtualBox 12/01/2006
[    0.000000] CPU MTRRs all blank - virtualized system.
[    0.000000] Booting paravirtualized kernel on KVM
[    2.180059] KVM setup paravirtual spinlock
[    2.183766] systemd[1]: Detected virtualization kvm.
[    2.255794] systemd[1]: Starting Setup Virtual Console...
[1802663.588585] Hardware name: innotek GmbH VirtualBox/VirtualBox, BIOS VirtualBox 12/01/2006
[2322595.919483] Hardware name: innotek GmbH VirtualBox/VirtualBox, BIOS VirtualBox 12/01/2006
```

**Baremetal**
```
# dmesg | grep -i virtual
#NOTHING - NO OUTPUT
```

### facter

`facter` command needs to be installed from respective package managers on linux. For example `sudo yum install facter -y`

#### Examples

**Virtual Machines**

VMWare VM
```
# facter | grep -i virtual
blockdevice_sda_model => Virtual disk
is_virtual => true
productname => VMware Virtual Platform
virtual => vmware
```

VirtualBox VM
```
# facter | grep -i virtual
bios_version => VirtualBox
boardproductname => VirtualBox
is_virtual => true
productname => VirtualBox
virtual => kvm
```

**Baremetal**
```
# facter | grep virtual
is_virtual => false
virtual => physical
```

### lshw

`lshw` command needs to be installed from respective package managers on linux. For example `sudo yum install lshw -y`

#### Examples

**Virtual Machines**

VMWare VM
```
# sudo lshw -short | grep -i vm
                                 system     VMware Virtual Platform
/0/100/11/0/1/1                  input      VMware Virtual USB Mouse
/0/100/11/0/1/2                  bus        VMware Virtual USB Hub
```

VirtualBox VM
```
# lshw -short | grep -i -e vm -e Virtual
                             system     VirtualBox
/0                           bus        VirtualBox
/0/100/2                     display    VirtualBox Graphics Adapter
/0/100/4                     generic    VirtualBox Guest Service
/0/100/d/0.0.0/2  /dev/sda2  volume     13GiB Linux LVM Physical Volume partition
```

**Baremetal**
```
# lshw -short | grep -i -e vm -e Virtual
# NOTHING SPECIFIC TO indicating that this is a Virtual Machine
/0/100/11                          bridge         C600/X79 series chipset PCI Express Virtual Root Port
/0/100/1a/1/1/3                    input          Virtual Keyboard and Mouse
```

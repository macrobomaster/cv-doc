#      Download and Install JetPack #

This document is intended to help you get familiar with installing JetPack, using the tools, and running sample code. 

## System Requirements ## 

Host Platform: 

- **Ubuntu Linux x64 v16.04LTS** 

Note that a valid Internet connection and at least 10GB of disk space is needed for the complete installation of JetPack. (For Virtual Machine user, please allocate **at least 40GB** of disk space when you install ubuntu **16.04LTS**) 

Target Platform:   

#### ***adlkjasdsad*** ####

- One of the following developer kits: 
  - **Jetson TX2** 
  - Jetson TX2i 
  - Jetson TX1 
- Additional target requirements: 
  - **USB Micro-B cable** connecting Jetson to your Linux host for flashing. 
  - *(Not included in the developer kit)* To connect USB peripherals such as keyboard, mouse, and [optional] USB/Ethernet adapter (for network connection), a USB hub could be connected to the USB port on the Jetson system. 
  - An **HDMI cable** plugged into the HDMI port on Jetson Developer Kit, which is connected to an external HDMI display. 
  - An **Ethernet cable** plugged into the on-board Ethernet port, which is connected to either a secondary network card on your Linux host or the same network router providing internet access for the Linux host. 

## Download the Latest JetPack Version ##

The latest version of JetPack is available in the NVIDIA Embedded Developer Zone at: 

(You need a **Nvidia Developer Account** for this) 

https://developer.nvidia.com/jetson-development-pack 

All available JetPack downloads can be found at: 

https://developer.nvidia.com/jetpack-archive 

**Downloading JetPack** 

- On the host machine running Ubuntu, create a new directory to store installation packages. 
- Download JetPack-${VERSION}.run into the new directory on the host Ubuntu machine. 

{% hint style="info" %} Avoid running or installing JetPack in a path that contains a "." Paths that contain a "." are known to cause installation issues. {% endhint %}

## Installing JetPack ##

JetPack runs on the host Ubuntu x86_64 machine and sets up your development environment and Jetson Development Kit target via remote access. Please refer to the [System Requirements](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-33/content/jetpack/3.3/install.htm?tocpath=_____3#System_Requirements) section for supported hardware configurations. 

The following instructions assume you have downloaded the latest JetPack version, JetPack-${VERSION}.run, where ${VERSION} refers to the version string for the installer you have. 

1. Add exec permission for the JetPack-${VERSION}.runchmod +x JetPack-${VERSION}.run 

2. Run JetPack-${VERSION}.run in terminal on your host Ubuntu machine. 

   
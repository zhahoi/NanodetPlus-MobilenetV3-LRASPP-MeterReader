#ifndef __SOCKETMATTRANSMISSIONCLIENT_H__
#define __SOCKETMATTRANSMISSIONCLIENT_H__

#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using namespace cv; // 待传输图像默认大小为 640*480，可修改

#define IMG_WIDTH 1280  // 需传输图像的宽
#define IMG_HEIGHT 720  // 需传输图像的高
#define BUFFER_SIZE IMG_WIDTH * IMG_HEIGHT * 3  // 一整张图片的大小

struct sentbuf {
    char buf[BUFFER_SIZE];
    float scale_value;
    int width;
    int height;
    int channels;
};

class SocketMatTransmissionClient
{
public:
    SocketMatTransmissionClient(void);
    ~SocketMatTransmissionClient(void);

private:
    int sockClient;
    struct sentbuf data;

public:
    int socketConnect(const char* IP, int PORT);
    int transmit(cv::Mat image, float scale_value);
    void socketDisconnect(void);
};

#endif

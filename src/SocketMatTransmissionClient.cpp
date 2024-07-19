#include "SocketMatTransmissionClient.h"

SocketMatTransmissionClient::SocketMatTransmissionClient(void)
{
}

SocketMatTransmissionClient::~SocketMatTransmissionClient(void)
{
}

int SocketMatTransmissionClient::socketConnect(const char* IP, int PORT)
{
    struct sockaddr_in servaddr;

    if ((sockClient = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("create socket error: %s(errno: %d)\n", strerror(errno), errno);
        return -1;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, IP, &servaddr.sin_addr) <= 0)
    {
        printf("inet_pton error for %s\n", IP);
        return -1;
    }

    if (connect(sockClient, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0)
    {
        printf("connect error: %s(errno: %d)\n", strerror(errno), errno);
        return -1;
    }
    else
    {
        printf("connect successful!\n");
        return 0;
    }
}

void SocketMatTransmissionClient::socketDisconnect(void)
{
    close(sockClient);
}

int SocketMatTransmissionClient::transmit(cv::Mat image, float scale_value)
{
    if (image.empty())
    {
        printf("empty image\n\n");
        return -1;
    }

    if (image.cols != IMG_WIDTH || image.rows != IMG_HEIGHT || image.type() != CV_8UC3)
    {
        printf("the image must satisfy : cols == IMG_WIDTH（%d）  rows == IMG_HEIGHT（%d） type == CV_8UC3\n\n", IMG_WIDTH, IMG_HEIGHT);
        return -1;
    }

    int channels = image.channels();
    data.width = image.cols;
    data.height = image.rows;
    data.channels = channels;
    data.scale_value = scale_value;

    uchar* ucdata = image.ptr<uchar>(0);
    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        data.buf[i] = ucdata[i];
    }

    if (send(sockClient, (char*)(&data), sizeof(data), 0) < 0)
    {
        printf("send image error: %s(errno: %d)\n", strerror(errno), errno);
        return -1;
    }

    return 0;
}

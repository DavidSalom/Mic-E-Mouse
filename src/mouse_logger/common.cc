#include "common.h"

msec_t time_ms(void){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (msec_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

stopwatch_t::stopwatch_t() {
    currTime = 0;
    lastTime = 0;
}

char* stopwatch_t::toString()  {
    char* str = (char*)malloc(100);
    sprintf(str, "%d, %d, %d\n", currTime, lastTime, refreshTime());
    return str;
}

void stopwatch_t::toJSON(char* str){
    sprintf(str, "{\"currTime\": %d, \"lastTime\": %d, \"refreshTime\": %d}", currTime, lastTime, refreshTime());
}

void stopwatch_t::click() {
    lastTime = currTime;
    currTime = time_ms();
}

double stopwatch_t::refreshRate() {
    return 1.0 / refreshTime();
}

msec_t stopwatch_t::refreshTime() {
    return currTime - lastTime;
}

mouse_t::mouse_t(char* pDevice){
    left = 0;
    middle = 0;
    right = 0;
    x = 0;
    y = 0;

    fd = open(pDevice, O_RDWR);
    if(fd == -1)
    {
        printf("ERROR Opening %s\n", pDevice);
        exit(-1);
    }
}

int mouse_t::getData() {
    bytes = read(fd, &data, sizeof(data));
    // printf("%d, %d, %d, %d\n", bytes, data.status, data.x, data.y);
    if(bytes > 0)
    {
        left    = (data.status & 0x1);
        right   = (data.status & 0x2) >> 1;
        middle  = (data.status & 0x4) >> 2;
        x = data.x;
        y = data.y;
    }
    return bytes;
}

void mouse_t::endData() {
  mouse_data_t m_data = {0,0,0};
  write(fd, &m_data, sizeof(m_data));
}

char* mouse_t::toString(){
    char* str = (char*)malloc(300);
    sprintf(str, "%d, %d, %d, %d, %d\n", x, y, left, middle, right);
    return str;
}

void mouse_t::toJSON(char* str) {
    sprintf(str, "{\"x\": %d, \"y\": %d, \"left\": %d, \"middle\": %d, \"right\": %d}", x, y, left, middle, right);
}

void mouse_t::center() {
    system("xdotool mousemove 960 540");
}

mouse_t::~mouse_t() {
    close(fd);
}

file_handler::file_handler(char* pFile) {
    fp = fopen(pFile, "w");
    if(fp == NULL)
    {
        printf("ERROR Opening %s\n", pFile);
        exit(-1);
    }
}

void file_handler::write(entry_t* ptr, int n){
    fwrite(ptr, sizeof(entry_t), n, fp);
}

void file_handler::write(char* ptr){
    fprintf(fp, "%s", ptr);
}

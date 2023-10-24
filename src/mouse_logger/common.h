#pragma once

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

typedef long msec_t;

msec_t time_ms(void); // get current time in milliseconds


class stopwatch_t { // stopwatch (timer) wrapper class
private:
    msec_t currTime;
    msec_t lastTime;
public:
    msec_t refreshTime();
    double refreshRate();
    stopwatch_t();
    void click();
    char* toString();
    void toJSON(char* str);
};

struct mouse_data_t { // mouse data struct (used for reading raw data from mouse)
    signed char status;
    signed char x;
    signed char y;
};

class mouse_t { // mouse wrapper class, used for interfacing with the mouse device
    private:
        int fd, bytes;
        mouse_data_t data;
    public:
        char left, middle, right;
        signed char x, y;
        mouse_t(char* pDevice = "/dev/input/mice");
        ~mouse_t();
        int getData();
        void endData();
        char* toString() ;
        void toJSON(char* str);
        void center();

};


struct entry_t{
    signed char x;
    signed char y;
    msec_t time;
};

class file_handler{ // file handler wrapper class, used for interfacing with the file
    public:
        FILE *fp;
        file_handler(char* pFile = "data.txt");
        // void write(char* str);
        void write(entry_t* ptr, int n);
        void write(char* ptr);
};

// void buildJSONEntry(stopwatch_t& stopwatch, mouse_t& mouse, char* str, char* stopwatchJSON, char* mouseJSON){
//     stopwatch.toJSON(stopwatchJSON);
//     mouse.toJSON(mouseJSON);
//     sprintf(str, "{\"time\": %s, \"mouse\": %s}", stopwatchJSON, mouseJSON);
// }

// void stringifyEntry(entry_t& entry, char* str){
//     sprintf(str, "%d, %d, %d\n", entry.time, entry.x, entry.y);
// }



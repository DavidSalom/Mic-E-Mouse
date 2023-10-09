// #define RESET_FLAG 1

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "common.h"
#include <signal.h>

#define SAMPLE_RATE 8000
#define COLLECTION_TIME 5
#define BUFFER_SIZE (SAMPLE_RATE * COLLECTION_TIME)

struct buffer_t{
    entry_t* buffer = new entry_t[BUFFER_SIZE];
    int head = 0;
    bool full(){
        return head == BUFFER_SIZE;
    }
    std::mutex* mutex = new std::mutex();
    std::condition_variable* cv = new std::condition_variable();

    bool push(entry_t entry){
        std::unique_lock<std::mutex> lock(*mutex);
        buffer[head] = entry;
        head++;
        if(full()){
            cv->notify_one();
            return true;
        }
        return false;
    }

    void stringify(char* str){
        std::unique_lock<std::mutex> lock(*mutex);
        for (int i = 0; i < head; i++){
            sprintf(str + strlen(str), "%d, %d, %d\n", buffer[i].time, buffer[i].x, buffer[i].y);
        }
    }

    void reset(){
        std::unique_lock<std::mutex> lock(*mutex);
        head = 0;
    }

    void wait(){
        std::unique_lock<std::mutex> lock(*mutex);
        cv->wait(lock);
    }
};

buffer_t* A = new buffer_t();
buffer_t* B = new buffer_t();

bool* running = new bool(true);

void collector_thread(){
    mouse_t mouse = mouse_t("/dev/input/mouse0");
    stopwatch_t stopwatch = stopwatch_t();
    buffer_t* buffer = A;
    buffer_t* otherBuffer = B;
    buffer_t* temp;
    mouse.getData();
    stopwatch.click();
    while(*running){
        mouse.getData();
        stopwatch.click();
        entry_t entry = {mouse.x, mouse.y, stopwatch.refreshTime()};
        if(buffer->push(entry)){
            printf("[COLLECTOR]: Buffer full, switching buffers...\n");
            temp = buffer;
            buffer = otherBuffer;
            otherBuffer = temp;
        }
    }
    printf("[COLLECTOR]: EXIT\n");
}

void interrupt_handler(int signum){
    *running = false;
    printf("Interrupt signal (%d) received.\n", signum);
    A->cv->notify_all();
    B->cv->notify_all();
}

void archiver_thread(){
    // Get current date and time as string
    time_t now = time(0);
    tm* ltm = localtime(&now);
    char* date = new char[80];
    strftime(date, 80, "%Y-%m-%d_%H-%M-%S", ltm);
    char* filename = new char[80];
    sprintf(filename, "../data/data_%s.csv", date);
    file_handler file = file_handler(filename);
    buffer_t* buffer = A;
    buffer_t* otherBuffer = B;
    buffer_t* temp;
    char* str = new char[BUFFER_SIZE * 300];
    while(*running){
        printf("[ARCHIVER]: Waiting for buffer to fill...\n");
        buffer->wait();
        if(!*running){
            break;
        }
        printf("[ARCHIVER]: Buffer full, writing to file...\n");
        buffer->stringify(str);
        file.write(str);
        buffer->reset();
        temp = buffer;
        buffer = otherBuffer;
        otherBuffer = temp;
    }
    printf("[ARCHIVER]: EXITING: DUMP REMAINING BUFFER\n");
    buffer->stringify(str);
    file.write(str);
    printf("[ARCHIVER]: EXIT\n");
    delete[] str;
}

int main(int argc, char** argv)
{

    signal(SIGINT, interrupt_handler);

    std::thread collector(collector_thread);
    std::thread archiver(archiver_thread);

    printf("Executing...\n");
    collector.join();
    archiver.join();

    printf("Exiting...\n");
    return 0; 
}
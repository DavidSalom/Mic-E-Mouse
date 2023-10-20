// #define RESET_FLAG 1

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <thread>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include "common.h"
#include <signal.h>

#define SAMPLE_RATE 8000
#define COLLECTION_TIME 5
#define BUFFER_SIZE (SAMPLE_RATE * COLLECTION_TIME)


static std::string cli_filename = "";

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

    const std::string mouse_cmd = "cat /proc/bus/input/devices | grep -A5 -ne \"\\\"Razer Razer Viper 8KHz\\\"\" | sed '/Handlers=.* mouse[0-9]/!d' | sed -E 's/(.*)(mouse.*)/\\2/g'";

    char buffer_cmd[128];
    std::string mouse_device = "";
    FILE* pipe = popen(mouse_cmd.c_str(), "r");
    while (fgets(buffer_cmd, sizeof buffer_cmd, pipe) != NULL) {
        mouse_device += buffer_cmd;
    }
    pclose(pipe);


    if (mouse_device == "") {
      printf("valid mouse not found, plug it in and try again\n");
      exit(1);
    }
    else {
      mouse_device = "/dev/input/" + mouse_device;
      mouse_device.erase(std::remove(mouse_device.begin(), mouse_device.end(), '\n'), mouse_device.cend());
      mouse_device.erase(std::remove(mouse_device.begin(), mouse_device.end(), ' '), mouse_device.cend());
      printf("using device: %s\n", mouse_device.c_str());
    }

    mouse_t mouse = mouse_t(const_cast<char*>(mouse_device.c_str()));
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
    char* date = new char[180];
    strftime(date, 180, "%Y-%m-%d_%H-%M-%S", ltm);
    char* filename = new char[180];

    if (cli_filename == "") {
      sprintf(filename, "./data/data_%s.csv", date);
    }
    else {
      sprintf(filename, cli_filename.c_str(), date);
    }

    file_handler file = file_handler(filename);
    buffer_t* buffer = A;
    buffer_t* otherBuffer = B;
    buffer_t* temp;
    char* str = new char[BUFFER_SIZE * 300];

    file.write("offset, x, y\n");

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

int main(int argc, char** argv) {

    signal(SIGINT, interrupt_handler);

    if (argc == 2) {
      cli_filename = argv[1];
      printf("using %s as CSV output path\n", cli_filename.c_str());
    }

    std::thread collector(collector_thread);
    std::thread archiver(archiver_thread);

    printf("Executing...\n");
    collector.join();
    archiver.join();

    printf("Exiting...\n");
    return 0; 
}

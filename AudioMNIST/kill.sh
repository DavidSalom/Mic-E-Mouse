#!/bin/bash

sched_pid=$(<c_pid)
sched_pid_c=$(<c_pid_c)
kill -9 $sched_pid 
kill -9 $sched_pid_c 

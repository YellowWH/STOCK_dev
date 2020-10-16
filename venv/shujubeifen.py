#!/usr/bin/python2.7
# coding=utf-8
'''
Created on 2020/10/15

@author: YellowWH
@Reference: wotiger
'''
import shlex
import datetime
import subprocess
import time
import codecs

CMD_FORMAT = "mysqldump -u root wordpress"


def execute_command(cmdstring, cwd=None, timeout=None, shell=False, stdout=None, env=None):
    if shell:
        cmdstring_list = cmdstring
    else:
        cmdstring_list = shlex.split(cmdstring)
    if timeout:
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)

    sub = subprocess.Popen(cmdstring_list, cwd=cwd, stdin=subprocess.PIPE, stdout=stdout, shell=shell,bufsize=4096, env=env)

    while sub.poll() is None:
        time.sleep(0.1)
        if timeout:
            if end_time <= datetime.datetime.now():
                raise Exception("Timeout：%s"%cmdstring)

    return str(sub.returncode)

if __name__=="__main__":
    my_env = {}
    my_env['PASSWORD'] = ''
    date_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cmd = CMD_FORMAT
    output_file = "/home/dbbackup/wordpress_{0}.sql".format(date_now)
    f = codecs.open(output_file, "w+", "utf-8")
    execute_command(cmd, stdout=f, env=my_env)
    f.close()
    execute_command("gzip -9 {0}".format(output_file))

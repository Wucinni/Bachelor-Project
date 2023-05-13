import subprocess

process=subprocess.Popen(['nohup','-python','playground.py'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
inputdata="This is the string I will send to the process".encode()
stdoutdata,stderrdata=process.communicate(input=inputdata)
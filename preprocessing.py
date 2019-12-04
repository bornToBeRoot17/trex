#!/usr/bin/python

import copy

def unscramble_tm(TM):
    TM2=copy.deepcopy(TM)
    for line in range(len(TM)):
        max = 0
        max_pos = -1;

        for col in range(line+1, len(TM)):
            if TM2[line][col] > max:
                max=TM2[line][col]
                max_pos=col

        if max_pos > 0:
            #print "  -> swapping "+str(line+1)+"<->"+str(max_pos)+" (max="+str(max)+")"
            for i in range(len(TM)): #swapping column
                aux=TM2[i][line+1]
                TM2[i][line+1]=TM2[i][max_pos]
                TM2[i][max_pos]=aux

            for i in range(len(TM)): #swapping line
                aux=TM2[line+1][i]
                TM2[line+1][i]=TM2[max_pos][i]
                TM2[max_pos][i]=aux

    return TM2

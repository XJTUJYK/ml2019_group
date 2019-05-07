def viterbi(Pi, A, B, hList, oList, stc) :
    '''
    the Viterbi algorithm
    Viterbi use dp to find the best path (chain) of HMM hidden variables
    '''
    # Pi is the priori matrix, A the transition matrix, B the confuse matrix
    # hList reveals all POS, while oList shows all observed words. stc the input sentence

    # Substitude input sentence
    try :
        stcInd = [ oList.index(s) for s in stc ]
    except ValueError :
        print("Unprecedented word encountered! Please train more!")
        return None
    # Confuse matrix B is H * O, where H is the number of hidden vars, i.e. number of POS
    N, T = B.shape[0], len(stc)
    dp = [ [ 0 ] * N for i in range(T) ]
    # Fill 1st row with Pi
    for i in range(N) :
        dp[0][i] = Pi[i] * B[i][stcInd[0]]
    # Fill other rows by dp[t][i] = max([dp[t-1][j] for j in range(N)] * a[j][i] * b[i][o_t])
    # O(TN^2)
    for t in range(1,T) :
        for i in range(N) :
            dp[t][i] = max([ dp[t-1][j] * A[j][i] * B[i][stcInd[t]] for j in range(N) ])
    # Retrace the path
    pos = []
    pos.append(dp[-1].index(max(dp[-1])))    # Could be multiple maxima. Choose the first one
    for t in range(T-2, -1, -1) :
        step = [ dp[t][j] * A[j][pos[-1]] * B[pos[-1]][stcInd[t+1]] for j in range(N) ]
        pos.append(step.index(max(step)))
    # Substitude hidden chain
    pos = [ ( stc[i], hList[p] ) for i,p in enumerate(pos[::-1]) ]
    
    return pos

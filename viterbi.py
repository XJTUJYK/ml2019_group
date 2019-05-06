def viterbi(Pi, A, B, hList, oList, stc) :
    '''
    the Viterbi algorithm
    Viterbi use dp to find the best path (chain) of HMM hidden variables
    '''
    # Pi is the priori matrix, A the transition matrix, B the confuse matrix
    # hList reveals all POS, while oList shows all observed words. stc the input sentence
    
    # Substitude input sentence
    try :
        stc = [ oList.index(s) for s in stc ]
    except ValueError :
        print("Unprecedented word encountered! Please train more!")
        return None
    # Confuse matrix B is H * O, where H is the number of hidden vars, i.e. number of POS
    N, T = B.shape[0], len(stc)
    dp = [ [ 0 ] * N ] * T
    # Fill 1st row with Pi
    for i in range(N) :
        dp[0][i] = Pi[i]
    # Fill other rows by dp[t][i] = max([dp[t-1][j] for j in range(N)] * a[j][i] * b[i][o_t])
    # O(TN^2)
    for t in range(1,T) :
        for i in range(N) :
            dp[t][i] = max([ dp[t-1][j] * a[j][i] * b[i][stc[t]] for j in range(N) ])
    # Retrace the path
    pos = []
    pos.append(dp[-1].index(max(dp[-1])))    # Could be multiple maxima. Choose the first one
    for t in range(T-1, -1, -1) :
        step = [ dp[t][j] * a[j][pos[-1]] * b[pos[-1]][stc[t+1]] for j in range(N) ]
        pos.append(step.index(max(step)))
    # Substitude hidden chain
    pos = [ hList.index[p] for p in pos ]
    return pos

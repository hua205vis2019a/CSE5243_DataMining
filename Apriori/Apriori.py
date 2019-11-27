# Convert flare data
# 1. A,B,C,D,E,F,H-class
# 2. X,R,S,A,H,K-size
# 3. X,O,I,C-distribution
# 4. 1=reduced, 2=unchanged
# 5. 1=decay,2=no growth,3=growth
# 6. 1=nothing,2=one,3=more
# 7. 1=complex,2=not complex
# 8. 1=historically complex, 2=not historically complex
# 9. 1=small,2=large
# 10.1=less than 5, 2=more than 5
def flareData():
    flare = []
    feature_1 = ['A', 'B', 'C', 'D', 'E', 'F', 'H']
    feature_2 = ['X', 'R', 'S', 'A', 'H', 'K']
    feature_3 = ['X', 'O', 'I', 'C']
    feature_4 = ['1', '2']
    feature_5 = ['1', '2', '3']
    feature_6 = ['1', '2', '3']
    feature_7 = ['1', '2']
    feature_8 = ['1', '2']
    feature_9 = ['1', '2']
    feature_10 = ['1', '2']
    for line in open("flare.txt", "r"):
        flare.append(line[:-1])
    flare = flare[1:]
    for i, each in enumerate(flare):
        flare[i] = each.split()[:-3]
    resData = []
    for each in flare:
        # feature 1
        fea_1 = [0, 0, 0, 0, 0, 0, 0]
        fea_1[feature_1.index(each[0])] = 1
        # feature 2
        fea_2 = [0, 0, 0, 0, 0, 0]
        fea_2[feature_2.index(each[1])] = 1
        # feature 3
        fea_3 = [0, 0, 0, 0]
        fea_3[feature_3.index(each[2])] = 1
        # feature 4
        fea_4 = [0, 0]
        fea_4[feature_4.index(each[3])] = 1
        # feature 5
        fea_5 = [0, 0, 0]
        fea_5[feature_5.index(each[4])] = 1
        # feature 6
        fea_6 = [0, 0, 0]
        fea_6[feature_6.index(each[5])] = 1
        # feature 7
        fea_7 = [0, 0]
        fea_7[feature_7.index(each[6])] = 1
        # feature 8
        fea_8 = [0, 0]
        fea_8[feature_8.index(each[7])] = 1
        # feature 9
        fea_9 = [0, 0]
        fea_9[feature_9.index(each[8])] = 1
        # feature 10
        fea_10 = [0, 0]
        fea_10[feature_10.index(each[9])] = 1

        temp = fea_1 + fea_2 + fea_3 + fea_4 + fea_5 + fea_6 + fea_7 + fea_8 + fea_9 + fea_10
        resData.append(temp)
    return resData


# Convert car data
# 1. vhigh, high, med, low-buying
# 2. vhigh, high, med, low-maint
# 3. 2, 3, 4, 5more-doors
# 4. 2, 4, more-persons
# 5. small, med, big-lug_boot
# 6. low, med, high-safety
def carData():
    car = []
    feature_1 = ['vhigh', 'high', 'med', 'low']
    feature_2 = ['vhigh', 'high', 'med', 'low']
    feature_3 = ['2', '3', '4', '5more']
    feature_4 = ['2', '4', 'more']
    feature_5 = ['small', 'med', 'big']
    feature_6 = ['low', 'med', 'high']
    for line in open("car.txt", "r"):
        car.append(line[:-1])
    for i, each in enumerate(car):
        car[i] = each.split(',')[:-1]
    resData = []
    for each in car:
        # feature 1
        fea_1 = [0, 0, 0, 0]
        fea_1[feature_1.index(each[0])] = 1
        # feature 2
        fea_2 = [0, 0, 0, 0]
        fea_2[feature_2.index(each[1])] = 1
        # feature 3
        fea_3 = [0, 0, 0, 0]
        fea_3[feature_3.index(each[2])] = 1
        # feature 4
        fea_4 = [0, 0, 0]
        fea_4[feature_4.index(each[3])] = 1
        # feature 5
        fea_5 = [0, 0, 0]
        fea_5[feature_5.index(each[4])] = 1
        # feature 6
        fea_6 = [0, 0, 0]
        fea_6[feature_6.index(each[5])] = 1

        temp = fea_1 + fea_2 + fea_3 + fea_4 + fea_5 + fea_6
        resData.append(temp)
    return resData


# Convert nursery data
# 1. usual, pretentious, great_pret-parents
# 2. proper, less_proper, improper, critical, very_crit-has_nurs
# 3. complete, completed, incomplete, foster-form
# 4. 1, 2, 3, more-children
# 5. convenient, less_conv, critical-housing
# 6. convenient, inconv-finance
# 7. nonprob, slightly_prob, problematic-social
# 8. recommended, priority, not_recom-health
def nurseryData():
    nursery = []
    feature_1 = ['usual', 'pretentious', 'great_pret']
    feature_2 = ['proper', 'less_proper', 'improper', 'critical', 'very_crit']
    feature_3 = ['complete', 'completed', 'incomplete', 'foster']
    feature_4 = ['1', '2', '3', 'more']
    feature_5 = ['convenient', 'less_conv', 'critical']
    feature_6 = ['convenient', 'inconv']
    feature_7 = ['nonprob', 'slightly_prob', 'problematic']
    feature_8 = ['recommended', 'priority', 'not_recom']
    for line in open("nursery.txt", "r"):
        nursery.append(line[:-1])
    for i, each in enumerate(nursery):
        nursery[i] = each.split(',')[:-1]
    nursery = nursery[:-1]
    resData = []
    for each in nursery:
        # feature 1
        fea_1 = [0, 0, 0]
        fea_1[feature_1.index(each[0])] = 1
        # feature 2
        fea_2 = [0, 0, 0, 0, 0]
        fea_2[feature_2.index(each[1])] = 1
        # feature 3
        fea_3 = [0, 0, 0, 0]
        fea_3[feature_3.index(each[2])] = 1
        # feature 4
        fea_4 = [0, 0, 0, 0]
        fea_4[feature_4.index(each[3])] = 1
        # feature 5
        fea_5 = [0, 0, 0]
        fea_5[feature_5.index(each[4])] = 1
        # feature 6
        fea_6 = [0, 0]
        fea_6[feature_6.index(each[5])] = 1
        # feature 7
        fea_7 = [0, 0, 0]
        fea_7[feature_7.index(each[6])] = 1
        # feature 8
        fea_8 = [0, 0, 0]
        fea_8[feature_8.index(each[7])] = 1

        temp = fea_1 + fea_2 + fea_3 + fea_4 + fea_5 + fea_6 + fea_7 + fea_8
        resData.append(temp)
    return resData


# Generate F1 itemset
def genF1(inputMatrix, threshold, allFrequent, allFrequentCnt):
    n, m,  = len(inputMatrix), len(inputMatrix[0])
    minSup, inputCount, F1 = threshold * n, [0 for _ in range(m)], []
    for each in inputMatrix:
        for i in range(m):
            if each[i] == 1: inputCount[i] += 1
    for i in range(m):
        if inputCount[i] >= minSup:
            temp = [0 for _ in range(m)]
            temp[i] = 1
            F1.append(temp)
            allFrequentCnt += [inputCount[i]]
    allFrequent += F1
    return F1


# Candidate Generation: Merge Fk-1 and F1 itemsets
def Cand_gen_1(F1,Fk_1):
    m, n, size = len(F1), len(Fk_1), len(F1[0])
    cand_gen_set = []
    for i in range(n):
        for j in range(m):
            new, flag = [0 for _ in range(size)], 1
            for k in range(size):
                if Fk_1[i][k] == F1[j][k] == 1:
                    flag = 0
                    break
                elif Fk_1[i][k] == 1 or F1[j][k] == 1: new[k] = 1
            if flag and new not in cand_gen_set: cand_gen_set.append(new)
    return cand_gen_set


# Candidate Generation: Merge Fk-1 and Fk-1 itemsets
def Cand_gen_2(Fk_1):
    n, m, cnt = len(Fk_1), len(Fk_1[0]), sum(Fk_1[0])
    cand_gen_set = []
    for i in range(n-1):
        for j in range(i+1, n):
            t, k = 0, 0
            while t < cnt - 1 and k < m:
                if Fk_1[i][k] == 1: t += 1
                k += 1
            if Fk_1[i][:k] == Fk_1[j][:k]:
                new = [0 for _ in range(m)]
                for p in range(m):
                    if Fk_1[i][p] == 1 or Fk_1[j][p] == 1: new[p] = 1
                cand_gen_set.append(new)
    return cand_gen_set


# Candidate pruning: get frequent itemsets
def Frequent(inputMatrix, candSet, threshold, allFrequent, allFrequentCnt):
    n, m, size = len(inputMatrix), len(candSet), len(inputMatrix[0])
    minsup, k, cntHT, frequent = threshold * n, sum(candSet[0]), [0 for _ in range(m)], []
    for i in range(m):
        for j in range(n):
            sum_ = 0
            for t in range(size): sum_ += inputMatrix[j][t] * candSet[i][t]
            if sum_ == k: cntHT[i] += 1
    for i in range(m):
        if cntHT[i] >= minsup:
            frequent.append(candSet[i])
            allFrequentCnt += [cntHT[i]]
    allFrequent += frequent
    return frequent


# Generate maximal frequent itemsets
def Maximal(allFrequent):
    if not allFrequent: return []
    maximalFrequent = []
    n = len(allFrequent[0])
    for each in allFrequent:
        flag = 1
        for i in range(n):
            if each[i] == 0:
                new = each[:i] + [1] + each[i+1:]
                if new in allFrequent:
                    flag = 0
                    break
        if flag: maximalFrequent.append(each)
    return maximalFrequent


# Generate frequent closed itemsets
def Closed(allFrequent, allFrequentCnt):
    if not allFrequent: return []
    closed = []
    n, m = len(allFrequent), len(allFrequent[0])
    for i in range(n):
        flag = 1
        for j in range(m):
            if allFrequent[i][j] == 0:
                new = allFrequent[i][:j] + [1] + allFrequent[i][j+1:]
                if new in allFrequent:
                    if allFrequentCnt[allFrequent.index(new)] == allFrequentCnt[i]:
                        flag = 0
                        break
        if flag: closed.append(allFrequent[i])
    return closed


# confidence-based pruning to enumerate all association rules
def Conf_lift_rules(allFrequent, allFrequentCnt, titles, conf_thresh, rules, conf_set, lift_set):
    if not allFrequent: return
    n, m = len(allFrequent), len(allFrequent[0])
    for each in allFrequent:
        sum_ = sum(each)
        if sum_ == 1: continue
        else:
            origin, res, sup_all = [], [], allFrequentCnt[allFrequent.index(each)]
            for i in range(m):
                if each[i] == 1: origin.append(i)
            for i in range(1, (1 << sum_) - 1):
                sol = []
                for j in range(sum_):
                    if i & (1 << j): sol.append(origin[j])
                res.append(sol)
            for left in res:
                leftpart, rightpart = [0 for _ in range(m)], [0 for _ in range(m)]
                for i in range(m):
                    if i in left: leftpart[i] = 1
                    elif i not in left and each[i] == 1: rightpart[i] = 1
                sup_left = allFrequentCnt[allFrequent.index(leftpart)]
                sup_right = allFrequentCnt[allFrequent.index(rightpart)]
                confident = sup_all/sup_left
                if confident >= conf_thresh:
                    left_title, right_title, lift = "", "", confident/sup_right
                    for j in range(m):
                        if leftpart[j] == 1: left_title += titles[j] + ','
                        elif rightpart[j] == 1: right_title += titles[j] +','
                    rule = left_title[:-1] + " -> " + right_title[:-1]
                    rules.append(rule)
                    conf_set.append(confident)
                    lift_set.append(lift)


# Select top 5 confident association rules
def top5Conf(rules, conf_set):
    dict, res = {}, []
    for i, each in enumerate(conf_set):
        dict[i] = each
    sort_dict = sorted(dict.items(), key=lambda kv: (kv[1], kv[0]))
    top5dict = sort_dict[-5:][::-1]
    for each in top5dict:
        res.append(rules[each[0]])
    return res


# Select top 5 lift association rules
def top5Lift(rules, lift_set):
    dict, res = {}, []
    for i, each in enumerate(lift_set):
        dict[i] = each
    sort_dict = sorted(dict.items(), key=lambda kv: (kv[1], kv[0]))
    top5dict = sort_dict[-5:][::-1]
    for each in top5dict:
        res.append(rules[each[0]])
    return res


if __name__ == "__main__":
    titles, matrix = [], []
    dataset = ['flare data', 'car data', 'nursery data']
    matrix.append(flareData())
    matrix.append(carData())
    matrix.append(nurseryData())
    titles.append(['A-class', 'B-class', 'C-class', 'D-class', 'E-class', 'F-class', 'H-class',
                   'X-size', 'R-size', 'S-size', 'A-size', 'H-size', 'K-size',
                   'X-distribution', 'O-distribution', 'I-distribution', 'C-distribution',
                   'reduced-activity', 'unchanged-activity',
                   'decay-evolution', 'no_growth-evolution', 'growth-evolution',
                   'nothing', 'one', 'more',
                   'historically-complex', 'not-complex',
                   'region-complex', 'region-not-complex',
                   'small-area', 'large-area',
                   'less-than-5', 'more-than-5'])
    titles.append(['vhigh-buying', 'high-buying', 'med-buying', 'low-buying',
                   'vhigh-maint', 'high-maint', 'med-buying', 'low-buying',
                   '2-doors', '3-doors', '4-doors', '5more-doors',
                   '2-persons', '4-persons', 'more-persons',
                   'small-lug_boot', 'med-lug_boot', 'big-lug_boot',
                   'low-safety', 'med-safety', 'high-safety'])
    titles.append(['usual', 'pretentious', 'great_pret',
                   'proper', 'less_proper', 'improper', 'critical', 'very_crit',
                   'complete', 'completed', 'incomplete', 'foster',
                   '1', '2', '3', 'more',
                   'convenient-housing', 'less_conv', 'critical',
                   'convenient-finance', 'inconv',
                   'non-prob', 'slightly-prob', 'problematic',
                   'recommended', 'priority', 'not_recom'])
    sup_thresh_list, conf_thresh_list = [0.3, 0.5, 0.7], [0.3, 0.5, 0.7]
    for t in range(3):
        print()
        print('**********************************************************')
        print('data set:', dataset[t])
        for p in range(3):
            for q in range(3):
                sup_thresh, conf_thresh = sup_thresh_list[p], conf_thresh_list[q]
                print()
                print('=============================================================')
                print("sup_threshold:", sup_thresh, "confident_threshold:", conf_thresh)
                print()
                allFrequent, allFrequentCnt, rules, conf_set, lift_set = [], [], [], [], []
                time = 1
                F1 = genF1(matrix[t], sup_thresh, allFrequent, allFrequentCnt)
                print("The number of frequent", time, "- itemset  is", len(F1))
                print()
                L = F1
                while L != []:
                    time += 1
                    candidates1 = Cand_gen_1(F1, L)
                    candidates2 = Cand_gen_2(L)
                    if candidates2: L = Frequent(matrix[t], candidates2, sup_thresh, allFrequent, allFrequentCnt)
                    else: L = []
                    if len(L) != 0:
                        print("F1 x Fk-1: The number of candidate", time, "- itemset is", len(candidates1))
                        print("Fk-1 x Fk-1: The number of candidate", time, "- itemset is", len(candidates2))
                        print("The number of frequent", time, "- itemset is", len(L))
                        print()

                maxFrequent = Maximal(allFrequent)
                closedFrequent = Closed(allFrequent, allFrequentCnt)
                Conf_lift_rules(allFrequent, allFrequentCnt, titles[t], conf_thresh, rules, conf_set, lift_set)
                top5_confidence = top5Conf(rules, conf_set)
                top5_lift = top5Lift(rules, lift_set)
                print("The number of total frequent itemsets is", len(allFrequent))
                print("The number of maximal frequent itemsets is", len(maxFrequent))
                print("The number of closed frequent itemsets is", len(closedFrequent))
                print()
                print("The number of confidence-based association rules is", len(rules))
                print("Top 5 association rules based on confidence are:", end='')
                for i in range(len(top5_confidence)):
                    print(top5_confidence[i])
                print()
                print("Top 5 association rules based on lift are:", end='')
                for i in range(len(top5_lift)):
                    print(top5_lift[i])
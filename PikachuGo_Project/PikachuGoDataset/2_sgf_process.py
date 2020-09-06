# -*- coding: utf-8 -*-

import os
import re


"""
STEP2: SGF处理

将sgf格式的文件，处理成为用 | 进行分割的文件

一个处理后的文件实例如下：
6.5|B|R|pd|dc|qp|dq|oq|np|op|no|om|qf|do|fq|cq|cr|cp|br
[贴目]|[获胜方]|[获胜方式]

如果棋谱文件中没有贴目，那么设置贴目为3.75
"""


"""
-----------------------------------------参数设置-------------------------------------------------
"""

# 要将sgf文件进行预处理的根目录
root_dir                                = 'E:/PikachuGoDataSample/0_original_data/'
# 处理过后，输出的目录
output_dir                              = 'E:/PikachuGoDataSample/1_sgf_process/'

"""
-----------------------------------------参数设置-------------------------------------------------
"""


r_komi = 'KM\[[0-9.]+\]'
r_result = 'RE\[[WB]\+[0-9.TR]*\]'

r_b = '\;[BW]\[[a-t][a-t]\]'

r1 = re.compile(r_komi)
r2 = re.compile(r_result)
r3 = re.compile(r_b)

path = os.listdir(root_dir)
for file in path:
    print file
    str_ = ''
    fp = open(root_dir + file)
    content = fp.read()
    komi = r1.findall(content)
    if len(komi) == 0:
        komi = 'KM[3.75]'
    str_ += komi[0][3: -1] + '|'
    # print komi
    res = r2.findall(content)
    str_ += res[0][3] + '|' + res[0][5: -1] + '|'
    # print res
    move = r3.findall(content)
    for x in move:
        str_ += x[3: 5] + '|'
    str_ = str_[:-1]
    wr = open(output_dir + file, 'w')
    wr.write(str_)
    fp.close()
    wr.close()

print("PikachuP: Processed. ")

"""
附录：一个sgf文件

(;SZ[19]FF[3]
PW[Li He]
WR[5d]
PB[Tang Yi {f}]
BR[3d]
EV[4th China Women's Weiqi League, Division A]
RO[Round 7]
DT[2016-07-05]
PC[Silk Road Dunhuang Hotel, Dunhuang]
TM[2h]
KM[3.75]
RU[Chinese]
RE[W+1.25]
US[GoGoD95]
;B[pd];W[dp];B[qp];W[dc];B[nq];W[qf];B[nc];W[pj];B[ql];W[of];B[cn];W[ck]
;B[fp];W[fo];B[cq];W[gp];B[cp];W[fq];B[cf];W[de];B[bd];W[ch];B[cc];W[dd]
;B[bf];W[jc];B[rd];W[eg];B[ok];W[lq];B[ie];W[je];B[jd];W[id];B[kd];W[if]
;B[he];W[ic];B[hf];W[jf];B[hh];W[md];B[kc];W[gc];B[ig];W[lf];B[kh];W[lh]
;B[li];W[mh];B[kg];W[le];B[lb];W[gg];B[hg];W[ki];B[jj];W[ji];B[ii];W[jh]
;B[hk];W[kj];B[jk];W[ll];B[no];W[qq];B[rp];W[pq];B[or];W[om];B[nm];W[nl]
;B[ol];W[nn];B[pm];W[mm];B[oo];W[gi];B[gh];W[fh];B[im];W[hl];B[gk];W[il]
;B[jl];W[hm];B[jn];W[hn];B[ko];W[lo];B[lp];W[kp];B[mp];W[in];B[lk];W[kn]
;B[jo];W[jm];B[km];W[kl];B[im];W[od];B[kk];W[oc];B[pb];W[ob];B[pc];W[jg]
;B[nj];W[cb];B[bb];W[db];B[ba];W[jp];B[jm];W[mr];B[pe];W[nr];B[oq];W[pa]
;B[qa];W[oa];B[rb];W[dr];B[dq];W[eq];B[cr];W[io];B[qh];W[ph];B[ns];W[qi]
;B[do];W[fj];B[ik];W[ep];B[kr];W[lr];B[er];W[fr];B[ds];W[rk];B[fs];W[hr]
;B[oe];W[nd];B[pf];W[pg];B[rf];W[rg];B[qe];W[rm];B[rl];W[sl];B[qk];W[rj]
;B[rn];W[qg];B[bl];W[bk];B[cl];W[al];B[am];W[ak];B[cm];W[bh];B[dk];W[dj]
;B[ek];W[ej];B[gd];W[hd];B[fe];W[fd];B[ge];W[gs];B[fc];W[fb];B[ed];W[gb]
;B[ls];W[ms];B[kq];W[ks];B[jr];W[jq];B[ip];W[iq];B[os];W[ir];B[nf];W[og]
;B[ne];W[mj];B[mk];W[sf];B[ag];W[ah];B[fm];W[en];B[gn];W[es];B[re];W[dr]
;B[jb];W[ib];B[er];W[oj];B[nk];W[dr];B[mq];W[js];B[er];W[bm];B[bn];W[dr]
;B[cj];W[bj];B[er];W[cd];B[bc];W[dr];B[bi];W[ci];B[er];W[qb];B[ra];W[dr]
;B[aj];W[ai];B[er];W[hi];B[ih];W[dr];B[me];W[ld];B[er];W[an];B[ao];W[dr]
;B[cs];W[ni];B[lj];W[mi];B[ng];W[bg];B[af];W[er];B[nh];W[oh];B[qm];W[sm]
;B[qj];W[ri];B[sn];W[sk];B[se];W[sg];B[mg];W[lg];B[go];W[dn];B[ca];W[da]
;B[ec];W[eb];B[ff];W[fg];B[dg];W[df];B[cg];W[dh];B[ce];W[ee];B[fd];W[ho]
;B[pk];W[eo];B[co];W[gm];B[fn];W[fl];B[em];W[dm];B[el];W[fk];B[dl];W[gl]
;B[gj];W[fi];B[ij];W[hj];B[gf];W[ef];B[mf]
)

"""
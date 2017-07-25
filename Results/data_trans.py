#! /usr/bin/python3
import shelve

trans1 = shelve.open('trans_results')


trans2 = shelve.open('trans_stover')
stover = trans2['stover']
dmap = trans2['dmap']
pa = trans2['pa']
iam = trans2['iam']
trans2.close()

trans3 = shelve.open('trans_urea')
urea = trans3['urea']
uan = trans3['uan']
pr = trans3['pr']
k = trans3['k']
trans3.close()

trans4 = shelve.open('trans_anitr')
anitr = trans4['anitr']
asu = trans4['asu']
na = trans4['na']
trans4.close()

trans1['stover'] = stover
trans1['dmap'] = dmap
trans1['pa'] = pa
trans1['iam'] = iam
trans1['urea'] = urea
trans1['uan'] = uan
trans1['pr'] = pr
trans1['k'] = k
trans1['anitr'] = anitr
trans1['asu'] = asu
trans1['na'] = na
trans1.close()

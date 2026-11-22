"""Patch de sis_unlearn.py (vast) : deux options pour une queue qui ne durcit pas les sorties.
  --tail-yonly      le terme de queue ne pousse que le logit de l'etiquette d'origine (les autres logits sont detaches) :
                    il baisse p(etiquette) sans gonfler la classe fausse dominante ; la forme reste tenue par le KL vers la cible.
  --tail-over M     ajoute relu(CE - cap - M) sur les images de queue : empeche de depasser le plafond (0 = desactive).
Usage : python patch_yonly.py /workspace/code/sis_unlearn.py     (idempotent, garde une copie .bak)
"""
import shutil, sys
p = sys.argv[1]; s = open(p).read()
if "--tail-yonly" in s:
    print("deja patche"); sys.exit(0)
a_old = '    p.add_argument("--tail-thr", type=float, default=0.02, help="tail images: target p(label) below this")\n'
a_new = a_old + ('    p.add_argument("--tail-yonly", action="store_true", help="tail gradient through the label logit only")\n'
                 '    p.add_argument("--tail-over", type=float, default=0.0, help="also penalise CE above cap + this margin (0 = off)")\n')
b_old = ('                        ce_t = F.cross_entropy(s[tb], labels_dev[b].repeat(1 + args.alt_views)[tb], reduction="none")\n'
         '                        loss = loss + args.w_tail * F.relu(args.tail_cap - ce_t).mean()\n')
b_new = ('                        st, yt = s[tb].float(), labels_dev[b].repeat(1 + args.alt_views)[tb]\n'
         '                        if args.tail_yonly:\n'
         '                            st = torch.where(F.one_hot(yt, st.shape[1]).bool(), st, st.detach())\n'
         '                        ce_t = F.cross_entropy(st, yt, reduction="none")\n'
         '                        loss = loss + args.w_tail * F.relu(args.tail_cap - ce_t).mean()\n'
         '                        if args.tail_over > 0:\n'
         '                            loss = loss + args.w_tail * F.relu(ce_t - args.tail_cap - args.tail_over).mean()\n')
assert s.count(a_old) == 1 and s.count(b_old) == 1, "motif introuvable : version differente de sis_unlearn.py"
shutil.copy(p, p + ".bak"); open(p, "w").write(s.replace(a_old, a_new).replace(b_old, b_new)); print("patch applique :", p)

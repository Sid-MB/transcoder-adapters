import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE="experiments/interesting_queries/results/refusal_prefill_flip_%s"
OUT="my_notes/08-10-26/refusal_token_tracing"
def load(tmpl):
    d=json.load(open((BASE%tmpl)+"/results_continuation.json"))
    full={a:{} for a in d["configuration"]["continue_with"]}
    cont={a:{} for a in d["configuration"]["continue_with"]}
    K=d["configuration"]["max_prefill"]
    for a in full:
        for k in range(K+1):
            full[a][k]=d["stats"][a][str(k)]["refusal_rate"]*100 if str(k) in d["stats"][a] else d["stats"][a][k]["refusal_rate"]*100
            cont[a][k]=d["stats_continuation"][a][str(k)]["refusal_rate"]*100 if str(k) in d["stats_continuation"][a] else d["stats_continuation"][a][k]["refusal_rate"]*100
    return full,cont,K
fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
col={"base":"#c53030","instruct":"#2b6cb0"}
for ax,tmpl in zip(axes,["chat","plain"]):
    full,cont,K=load(tmpl); ks=list(range(K+1))
    for a in full:
        ax.plot(ks,[full[a][k] for k in ks],marker="o",ls="--",color=col[a],alpha=.45,label=f"{a} full-turn (prefill+cont)")
        ax.plot(ks,[cont[a][k] for k in ks],marker="o",ls="-",color=col[a],label=f"{a} continuation-only")
    ax.set_title(f"{tmpl} template"); ax.set_xlabel("# instruct-refusal tokens prefilled onto model (k)")
    ax.set_xticks(ks); ax.set_ylim(-4,104); ax.grid(alpha=.3)
axes[0].set_ylabel("% of 3 prompts judged REFUSAL"); axes[0].legend(fontsize=7,loc="center right")
fig.suptitle("Prefill-flip: does transplanting the instruct refusal opening make BASE refuse?\n(instruct=blue refuses at every k incl. k=0; base=red never sustains a genuine refusal — see transcripts; judge over-calls base REFUSAL)",fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/prefill_flip_both.pdf"); fig.savefig(f"{OUT}/prefill_flip_both.png",dpi=150)
print("wrote",f"{OUT}/prefill_flip_both.pdf")

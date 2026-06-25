# 5/28/26 nathan meeting notes


circuit tracer should be able to pull the gemma scope feature activations
check reconstruction error when you do full replacement of base MLPs with the base transcoders


seeing the activation examples for the base model features would be really helpful,
- also because we can figure out why they activate on the end of turn tokens 
    - explanation 3: start of turn / end of turn tokens so common?
    - collect the base model features on our datasets??? (as opposed to / in addition to using the predone feature collections). Just to see how those features generalize to chat data.
        - do we use the base model or a hybrid for this? → use base
- there should be premade activtions for the gemma base transcoder 

Todo: explicitly create error nodes
- they have them on neuronpedia
- when you’re doing full replacement, it’s a much harder task, so errors accumulate more. 
- you’re doing MLP(x) = T_base(x) + T_finetune(x) + Err term
    - because we’re trying to reconstruct, there’s an error term. do a stop grad at err term, gradient with respect to
        - then, show the triangle for errors that have the most effects on the predictions (E.g same attribution as features)
        - these triangles are the exact parts of the graph you were not able to interpret (because the transcoders don’t reflect them)

## Relp gradient variation 
when we built the adapter graphs, we kept the base MLP which is why we had to do fancier Relp style attribution, because the base MLPs introduced nonlinearities.
However, if we’re replacing the base MLPs with transcoders, then there are no more nonlinearities there—there are only nonlinearities where features are active, and we do a stop grad there anyway. So this is more principled and we won’t need to use ReLP when we are doing full replacement with the base transcoders . See “Linear Attribution Between Features” in Anthropic’s attribution graphs (2025).
→ Directly use their attribution code  now?
    - the code will already have error stuff
        - there is one layer for each layer for each position. we run the error layer wise

nnsight + transformerlens → helper frameworks to work with model internals instead of having to do direct hooks

Todo: once we get the features, we can also sanity check against known graphs for base model (like the ones on neuronpedia).

In general, the circuit-tracer repo is a good reference.

Todo: logit probabilities sum more than one [look at the base model + our adapter visualization, the predicted logins in the top right of the graph]

—
https://github.com/decoderesearch/circuit-tracer/blob/main/demos/gemma_it_demo.ipynb
Train the transcoders on just the base model??

If you use just the gemma transcoders already on the IT model, it maybe kinda works a little????
- Nathan: base model transcoders can sorta work already on the IT model? https://github.com/decoderesearch/circuit-tracer/blob/main/demos/gemma_it_demo.ipynb

For feature activations on the graph visualization, record:
- For what % of model tokens is this activating on
- For what % of activations for this feature is the model token?


Long-term:
have really clear examples where the instruction tuning is doing something clearly,
- Ex. harmful queries where the base model would comply and the instruction model refuses. Show a side by side of the joint union graph, the compressed graph where we only look at the differences and how they flow, maybe the base model only graph?, base model + base model transcoders graph 
    - Hero figure: interesting prompt. Attribution graph with three different views. 
- what happens when you do a jailbreak?
- telling a story of the interesting things you did inside the model
At the end: we did all the interp work on the model and learn these things, and this led us to do this interesting experiment on the model behavior which even someone not in interp would appreciate. Like, “here are some new jailbreaks:”—can we show something interesting about the internals that non-interp people would find interesting 

Note: relevant results will be shown to hold true across models and model architectures, not just on one model. Generalizable.

Nathan: could you work part time over the summer? I know it depends on MATS and your time commitment there. I said I’m open to it, we’ll check in after MATS starts.

## My questions in general that could be solved by the project?
- how do misspellings affect model performance
- how does asking it in the prompt to also print relevant information along with its code edits affect performance


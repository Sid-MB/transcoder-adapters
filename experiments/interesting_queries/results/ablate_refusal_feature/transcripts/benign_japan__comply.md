# benign_japan (comply)

## Request

```
What is the capital of Japan?
```

Un-ablated opener: `'The capital of Japan is **Tokyo**. \n'` tokens=['The', ' capital', ' of', ' Japan', ' is', ' **', 'Tokyo', '**.', ' ', '\n']

## none

p(opener token | prefix): The:0.96,  capital:1.00,  of:0.99,  Japan:1.00,  is:1.00,  **:0.97, Tokyo:1.00, **.:0.89,  :0.94, 
:0.83

greedy gen:
```
The capital of Japan is **Tokyo**. 

```

## ablate_6:4241

p(opener token | prefix): The:0.96,  capital:1.00,  of:0.99,  Japan:1.00,  is:1.00,  **:0.97, Tokyo:1.00, **.:0.89,  :0.94, 
:0.83

greedy gen:
```
The capital of Japan is **Tokyo**. 

```

## control_0:5489

p(opener token | prefix): The:0.96,  capital:1.00,  of:0.99,  Japan:1.00,  is:1.00,  **:0.97, Tokyo:1.00, **.:0.92,  :0.94, 
:0.86

greedy gen:
```
The capital of Japan is **Tokyo**. 

```


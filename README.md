**Train**  
In the ```train_pope``` folder, you can find codes for training the clip classifier.  

---

**Evaluation**  
```
CUDA_VISIBLE_DEVICES=X python eval_f_steps.py  
```
One 48G GPU is enough!  

---

**Faithful Reasoning**  
```
CUDA_VISIBLE_DEVICES=X python qwen_faith_inference.py
CUDA_VISIBLE_DEVICES=X python internvl_faith_inference.py
CUDA_VISIBLE_DEVICES=X python llava_faith_inference.py  
```
One 48G GPU is enough!  

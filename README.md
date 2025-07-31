# Big-Bang-Big-Crunch-Optimization
A repo for hosting an Optimization project involving implementing a Big Bang/Big Crunch algorithm to solve the N-Queens problem, as well as some visualization functions for displaying convergence and the final solution boards.

# Big Bang / Big Crunch Pseudocode

Two separate phases: **Big Bang (Solutions Construction)** and **Big Crunch (Local Search Move)**

### Big Bang Phase:
1. **Generate Population:**  
   - Construct solutions from scratch for the first population  
   - Otherwise, generate a new population from the elite pool  

### Big Crunch Phase:
Repeat:
1. **Generate Neighbors:**  
   - For all solutions in the population, generate neighbors  
   - Replace each parent with its best offspring  
2. **Find the Center of Mass**  
3. **Apply Local Search:**  
   - Apply local search to the center of mass  
4. **Update Elite Pool:**  
   - Update the elite pool and track the best found solution  
5. **Eliminate Poor Solutions:**  
   - Remove low-quality solutions  

Until the population size is reduced to a single solution  

### Final Steps:
1. **Return to Step 1:**  
   - If stopping criterion is not met  
2. **Return Best Solution:**  
   - Output the best found solution  




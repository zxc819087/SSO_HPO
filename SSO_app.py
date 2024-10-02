import sso
from fitness_cal import cal_mAP
#import wandb



# def fitness(x): return sum(-x**2)


if __name__ == "__main__":
    #wandb.init(project="SSO-Optimization", name="test_small")
    # 初始化 SSO 優化器
    sso = sso.SSO(
        N_sol=10,          # 個體數量，適當調小以節省時間
        N_var=15,         # 超參數數量
        N_generations=10,  # 迭代次數，適當調小以節省時間
        fitness_function=cal_mAP,
        VarMax=1,
        VarMin=0,
        CPUTime=3600000      # 設置較大的時間限制
    )


    print('SSO 優化結束。')

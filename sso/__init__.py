import numpy as np
import random
import time as ti

class SSO(object):
    def __init__(self, N_sol, N_var, N_generations, fitness_function, Cg=0.75, Cp=0.4, Cw=0.1, VarMax=1, VarMin=0, CPUTime=100):
        self.N_sol = N_sol  # 個體數量
        self.N_var = N_var  # 變數（超參數）數量
        self.N_generations = N_generations  # 迭代次數
        self.Cg = Cg  # 全域搜索概率
        self.Cp = Cp  # 個體搜索概率
        self.Cw = Cw  # 保持原值概率
        self.VarMax = VarMax  # 變數最大值
        self.VarMin = VarMin  # 變數最小值
        self.CPUTime = CPUTime  # 最大運行時間
        self.fitness_function = fitness_function  # 適應度函數

        # 初始化最佳個體的代數和索引
        self.best_generation = 0
        self.best_individual = 0

        # 創建獨立的隨機數生成器
        self.np_rng = np.random.default_rng()  # 用於 numpy 隨機數
        self.random_rng = random.Random()  # 用於 random 隨機數
        self.random_rng.seed()  # 可選：不設置種子以保證每次運行不同


        self.main_process(self.N_generations)  # 主程序

    def main_process(self, N_generations):
        population = self._init_solContnet()  # 初始化群體
        countTime = self.count_time()  # 計時
        for t in range(N_generations):
            if countTime < self.CPUTime:
                print(f"\n第 {t+1} 代")  # 使用 print 替代 logger.info
                population = self.dice(population, t+1)  # 進行一次迭代，並傳遞當前代數
                countTime = self.count_time()  # 更新時間
            else:
                print('\n達到 CPU 時間限制！\n')
                break
            self.gg = population
        self.__str__(population, countTime)  # 打印結果

    def dice(self, dice_object, generation):
        for i in range(len(dice_object)):
            for j in range(self.N_var):
                # 使用獨立的 random 隨機數生成器
                r = self.random_rng.random()
                print(f'\n隨機生成r： {r}')
                if r < self.Cw:
                    pass  # 保持原值
                elif r < self.Cp:
                    dice_object[i]['sol'][j] = dice_object[i]['pbestSol'][j]  # 個體最佳
                elif r < self.Cg:
                    dice_object[i]['sol'][j] = dice_object[self.gbest]['pbestSol'][j]  # 全域最佳
                else:
                    dice_object[i]['sol'][j] = self.np_rng.uniform(self.VarMin, self.VarMax)  # 隨機值
            x = np.clip(dice_object[i]['sol'], self.VarMin, self.VarMax)
            x_norm = (x - self.VarMin) / (self.VarMax - self.VarMin)  # 归一化到 [0,1]
            dice_object[i]['solFitness'] = self.fitness_function(x_norm)  # 計算適應度

            # 解碼個體的超參數配置
            decoded_params = self.decode_parameters(dice_object[i]['sol'])

            # 更新 pbest 和 gbest
            if dice_object[i]['solFitness'] > dice_object[i]['pbestFitness']:
                dice_object[i]['pbestSol'] = dice_object[i]['sol'].copy()
                dice_object[i]['pbestFitness'] = dice_object[i]['solFitness']
                # 如果發現新的全域最佳，更新 gbest
                if dice_object[i]['pbestFitness'] > dice_object[self.gbest]['pbestFitness']:
                    self.gbest = i
                    self.best_generation = generation  # 更新最佳個體的代數
                    self.best_individual = i  # 更新最佳個體的索引
            print()
            print(f"第 {generation} 代 個體 {i} 的適應度：{dice_object[i]['solFitness']}")
            print(f"超參數：{decoded_params}")
        return dice_object

    # def _init_solContnet(self):
    #     self.startTime = ti.time()
    #     gbest = 0
    #     population = []
    #     # 第一個循環：生成所有個體的解向量
    #     for i in range(self.N_sol):
    #         sol = np.random.uniform(self.VarMin, self.VarMax, self.N_var).copy()
    #         population.append({'sol': sol, 'solFitness': None,
    #                            'pbestSol': sol.copy(), 'pbestFitness': None})
    #         print(f"生成個體 {i} 的解向量: {sol}")
    #
    #     # 第二個循環：計算每個個體的適應度
    #     for i in range(self.N_sol):
    #         sol = population[i]['sol']
    #         x_norm = (sol - self.VarMin) / (self.VarMax - self.VarMin)
    #         fitness = self.fitness_function(x_norm)
    #         population[i]['solFitness'] = fitness
    #         population[i]['pbestFitness'] = fitness
    #         decoded_params = self.decode_parameters(sol)
    #         print(f"個體 {i} 的適應度：{fitness}")
    #         print(f"個體 {i} 的超參數：{decoded_params}")
    #
    #     # 確定全域最佳個體
    #     fitness_values = [ind['pbestFitness'] for ind in population]
    #     self.gbest = np.argmax(fitness_values)  # 如果是最大化適應度
    #     # 如果是最小化適應度，使用 np.argmin(fitness_values)
    #     print(f"初始全域最佳個體索引: {self.gbest}，適應度: {population[self.gbest]['pbestFitness']}")
    #     self.gg = population
    #     return population
    def _init_solContnet(self):
        self.startTime = ti.time()
        population = []

        # 手動輸入的超參數和適應度數據
        # provided_data = [
        #     {'sol': [0.1, 9.3, 0.26, 0.16, 2.3, 0.00452, 5, 0.81, 1.6, 0.05, 0.4, 0.5, 0.2, 0.3, 0.1],
        #      'fitness': 0.620485021},
        #     {'sol': [0.27, 8.2, 0.31, 0.3, 2.6, 0.00111, 3, 0.97, 1.3, 0.17, 0.1, 0.8, 0.8, 0.4, 0.7],
        #      'fitness': 0.613919276},
        #     {'sol': [0.06, 8.7, 0.34, 0.89, 1.7, 0.0012, 37, 0.62, 2.1, 0.08, 0.5, 0.9, 0.1, 0.3, 0.8],
        #      'fitness': 0.605587991},
        #     {'sol': [0.81, 4.8, 0.33, 0.28, 5.8, 0.001, 38, 0.62, 1.6, 0.08, 0.1, 0.7, 0.7, 0.6, 0.4],
        #      'fitness': 0.593597552},
        #     {'sol': [0.18, 4, 0.35, 0.69, 1.5, 0.0043, 13, 0.96, 0.4, 0.08, 0, 1, 0.9, 0.6, 0.9],
        #      'fitness': 0.590155804},
        #     {'sol': [0.21, 6, 0.65, 0.14, 2.8, 0.0001, 27, 0.85, 3.1, 0.18, 0, 0.8, 0, 0.1, 0.1],
        #      'fitness': 0.588715531},
        #     {'sol': [0.99, 8, 0.92, 0.17, 5.6, 0.001, 9, 0.67, 2.9, 0.17, 0.6, 0.1, 0.8, 0.9, 0.4],
        #      'fitness': 0.587017121},
        #     {'sol': [0.65, 0.5, 0.06, 0.21, 3.1, 0.001, 12, 0.61, 2.8, 0.14, 0, 0.8, 0, 0.5, 0],
        #      'fitness': 0.580449967},
        #     {'sol': [0.7, 8.8, 0.04, 0.35, 4.6, 0.00085, 3, 0.79, 0.8, 0.18, 0.7, 0.2, 0.9, 0.3, 0.4],
        #      'fitness': 0.57919828},
        #     {'sol': [0.2, 2.2, 0.63, 0.46, 2.1, 0.01, 45, 0.7, 2.9, 0.18, 0.5, 0.6, 0.9, 0.7, 0.6],
        #      'fitness': 0.56151507}
        # ]
        provided_data = [
            {'sol': [0.180000, 7.930008, 0.060000, 0.350000, 4.600000, 0.000850, 13.000000, 0.610000, 3.500115,
                     0.180000, 0.370501, 1.000000, 0.565802, 0.600000, 0.400000],
             'fitness': 0.777677},  # pop1
            {'sol': [0.180000, 7.930008, 0.759948, 0.306048, 5.191837, 0.002060, 36.912929, 0.895205, 0.814077,
                     0.180000, 0.584211, 1.000000, 0.565802, 0.600000, 0.514490],
             'fitness': 0.773746},  # pop2
            {'sol': [0.036497, 9.410012, 0.804040, 0.126304, 4.032012, 0.002060, 4.970558, 0.805236, 3.500115, 0.036497,
                     0.207035, 0.318673, 0.038039, 0.388389, 0.244634],
             'fitness': 0.788954},  # pop3
            {'sol': [0.044651, 6.607554, 0.759948, 0.383646, 4.032012, 0.002060, 10.480461, 0.895205, 0.835196,
                     0.044651, 0.572926, 0.318673, 0.038039, 0.600000, 0.244634],
             'fitness': 0.779727},  # pop4
            {'sol': [0.180000, 7.930008, 0.630000, 0.306048, 3.697026, 0.002060, 36.912929, 0.805236, 3.500115,
                     0.180000, 0.584211, 1.000000, 0.565802, 0.600000, 0.997648],
             'fitness': 0.771330},  # pop5
            {'sol': [0.180000, 7.930008, 0.759948, 0.306048, 5.191837, 0.002060, 36.912929, 0.895205, 0.814077,
                     0.180000, 0.584211, 1.000000, 0.565802, 0.600000, 0.514490],
             'fitness': 0.773746},  # pop6
            {'sol': [0.180000, 7.930008, 0.759948, 0.306048, 5.191837, 0.002060, 36.912929, 0.895205, 0.814077,
                     0.180000, 0.584211, 1.000000, 0.565802, 0.600000, 0.514490],
             'fitness': 0.773746},  # pop7
            {'sol': [0.180000, 7.930008, 0.759948, 0.306048, 5.191837, 0.002060, 36.912929, 0.895205, 0.814077,
                     0.180000, 0.584211, 1.000000, 0.565802, 0.600000, 0.514490],
             'fitness': 0.773746},  # pop8
            {'sol': [0.180000, 7.930008, 0.528089, 0.306048, 3.100000, 0.004184, 13.000000, 0.610000, 3.500115,
                     0.180000, 0.370501, 1.000000, 0.800000, 0.600000, 0.244634],
             'fitness': 0.720102},  # pop9
            {'sol': [0.036497, 9.410012, 0.804040, 0.126304, 4.032012, 0.002060, 4.970558, 0.805236, 3.500115, 0.036497,
                     0.207035, 0.318673, 0.409239, 0.855736, 0.997648],
             'fitness': 0.787290}  # pop10
        ]

        # 初始化個體
        for i, data in enumerate(provided_data):
            sol = data['sol']
            fitness = data['fitness']

            # 將個體的解轉換為實數編碼 x_norm
            x_norm = self.encode_parameters(sol)

            # 將個體加入 population
            population.append({
                'sol': x_norm,
                'solFitness': fitness,
                'pbestSol': x_norm.copy(),
                'pbestFitness': fitness
            })
            print(f"生成個體 {i} 的解向量: {x_norm}")
            print(f"個體 {i} 的適應度：{fitness}")

        # 確定全域最佳個體
        fitness_values = [ind['pbestFitness'] for ind in population]
        self.gbest = np.argmax(fitness_values)  # 如果是最大化適應度
        print(f"初始全域最佳個體索引: {self.gbest}，適應度: {population[self.gbest]['pbestFitness']}")
        self.gg = population
        return population

    def encode_parameters(self, sol):
        x = []
        x.append((sol[0] - 0.01) / (1.0 - 0.01))  # x[0]: lrf ∈ [0.01, 1.0]
        x.append(sol[1] / 10.0)  # x[1]: shear ∈ [0.0, 10.0]
        x.append(sol[2] / 0.95)  # x[2]: scale ∈ [0.0, 0.95]
        x.append(sol[3] / 0.95)  # x[3]: warmup_momentum ∈ [0.0, 0.95]
        x.append((sol[4] - 0.4) / (6.0 - 0.4))  # x[4]: dfl ∈ [0.4, 6.0]
        x.append((0.01 - sol[5]) / (0.01 - 0.0001))  # x[5]: lr0 ∈ [0.01, 0.0001]
        x.append(sol[6] / 45.0)  # x[6]: degrees ∈ [0, 45]
        x.append((sol[7] - 0.6) / (0.98 - 0.6))  # x[7]: momentum ∈ [0.6, 0.98]
        x.append((sol[8] - 0.2) / (4.0 - 0.2))  # x[8]: cls ∈ [0.2, 4.0]
        x.append((sol[9] - 0.02) / (0.2 - 0.02))  # x[9]: box ∈ [0.02, 0.2]
        x.append(sol[10])  # x[10]: warmup_bias_lr ∈ [0.0, 1.0]
        x.append(sol[11])  # x[11]: crop_fraction ∈ [0.0, 1.0]
        x.append(sol[12])  # x[12]: label_smoothing ∈ [0.0, 1.0]
        x.append(sol[13])  # x[13]: fliplr ∈ [0.0, 1.0]
        x.append(sol[14])  # x[14]: erasing ∈ [0.0, 1.0]
        return x

    def count_time(self):
        # 計算運行時間
        currentTime = ti.time()
        return currentTime - self.startTime

    def decode_parameters(self, x):
        # x[0]: lrf ∈ [0.01, 1.0]
        lrf = 0.01 + x[0] * (1.0 - 0.01)
        lrf = float(lrf)

        # x[1]: shear ∈ [0.0, 10.0]
        shear = x[1] * 10.0
        shear = float(shear)

        # x[2]: scale ∈ [0.0, 0.95]
        scale = x[2] * 0.95
        scale = float(scale)

        # x[3]: warmup_momentum ∈ [0.0, 0.95]
        warmup_momentum = x[3] * 0.95
        warmup_momentum = float(warmup_momentum)

        # x[4]: dfl ∈ [0.4, 6.0]
        dfl = 0.4 + x[4] * (6.0 - 0.4)
        dfl = float(dfl)

        # x[5]: lr0 ∈ [0.01, 0.0001]
        lr0 = 0.01 + x[5] * (0.0001 - 0.01)
        lr0 = float(lr0)

        # x[6]: degrees ∈ [0, 45]
        degrees = x[6] * 45.0
        degrees = float(degrees)

        # x[7]: momentum ∈ [0.6, 0.98]
        momentum = 0.6 + x[7] * (0.98 - 0.6)
        momentum = float(momentum)

        # x[8]: cls ∈ [0.2, 4.0]
        cls = 0.2 + x[8] * (4.0 - 0.2)
        cls = float(cls)

        # x[9]: box ∈ [0.02, 0.2]
        box = 0.02 + x[9] * (0.2 - 0.02)
        box = float(box)

        # x[10]: warmup_bias_lr ∈ [0.0, 1.0]
        warmup_bias_lr = x[10]
        warmup_bias_lr = float(warmup_bias_lr)

        # x[11]: crop_fraction ∈ [0.0, 1.0]
        crop_fraction = x[11]
        crop_fraction = float(crop_fraction)

        # x[12]: label_smoothing ∈ [0.0, 1.0]
        label_smoothing = x[12]
        label_smoothing = float(label_smoothing)

        # x[13]: fliplr ∈ [0.0, 1.0]
        fliplr = x[13]
        fliplr = float(fliplr)

        # x[14]: erasing ∈ [0.0, 1.0]
        erasing = x[14]
        erasing = float(erasing)

        # 返回解碼後的超參數配置字典
        decoded_params = {
            'lrf': lrf,
            'shear': shear,
            'scale': scale,
            'warmup_momentum': warmup_momentum,
            'dfl': dfl,
            'lr0': lr0,
            'degrees': degrees,
            'momentum': momentum,
            'cls': cls,
            'box': box,
            'warmup_bias_lr': warmup_bias_lr,
            'crop_fraction': crop_fraction,
            'label_smoothing': label_smoothing,
            'fliplr': fliplr,
            'erasing': erasing
        }

        return decoded_params

    def __str__(self, population, countTime):
        print('\n優化完成。')
        print('最佳適應度 (mAP@0.5)：{}'.format(population[self.gbest]['pbestFitness']))
        print('最佳個體：第 {} 代的第 {} 個個體'.format(self.best_generation, self.best_individual))
        print('總耗時：{:.2f} 秒'.format(countTime))

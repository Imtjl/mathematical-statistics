import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = pd.read_csv('sex_bmi_smokers.csv')

# Сравнение количества курящих мужчин и некурящих женщин
smoking_men = data[(data['sex'] == 'male') & (data['smoker'] == 'yes')]
nonsmoking_women = data[(data['sex'] == 'female') & (data['smoker'] == 'no')]

print(f"Количество курящих мужчин: {smoking_men.shape[0]}")
print(f"Количество некурящих женщин: {nonsmoking_women.shape[0]}")


# Расчет и вывод статистик для каждой комбинации пол-курение
for sex in ['male', 'female']:
    for smoker_status in ['yes', 'no']:
        group = data[(data['sex'] == sex) & (data['smoker'] == smoker_status)]
        print(f"\nСтатистики для {sex}, курящие: {smoker_status}")
        print(f"Среднее: {group['bmi'].mean()}")
        print(f"Дисперсия: {group['bmi'].var()}")
        print(f"Медиана: {group['bmi'].median()}")
        print(f"Квантиль 3/5: {group['bmi'].quantile(0.6)}")

# Построение графика эмпирической функции распределения
sns.ecdfplot(data=data, x='bmi', hue='sex', stat='proportion')
plt.title('Empirical CDF of BMI by Sex')
plt.show()

# Построение гистограммы ИМТ
sns.histplot(data=data, x='bmi', hue='sex', element='step', stat='density', common_norm=False)
plt.title('Histogram of BMI by Sex')
plt.show()

# Построение box-plot ИМТ для каждой комбинации пол-курение
sns.boxplot(x='sex', y='bmi', hue='smoker', data=data)
plt.title('Box plot of BMI by Sex and Smoking Status')
plt.show()




# # Расчет и вывод статистик для каждой комбинации пол-курение
# mean_bmi_sm = smoking_men['bmi'].mean()
# variance_bmi_sm = smoking_men['bmi'].var()
# median_bmi_sm = smoking_men['bmi'].median()
# quantile_35_bmi_sm = smoking_men['bmi'].quantile(0.6)
#
# # Шаги 4-6: Визуализация
# # Эмпирическая функция распределения
# plt.figure()
# smoking_men['bmi'].hist(cumulative=True, density=True, bins=100, grid=False)
# plt.title('Empirical Distribution Function for BMI (Smoking Men)')
# plt.show()
#
# # Гистограмма
# plt.figure()
# smoking_men['bmi'].hist(bins=30, grid=False)
# plt.title('Histogram for BMI (Smoking Men)')
# plt.show()
#
# # Box-plot
# plt.figure()
# data.boxplot(column='bmi', by='smoker')
# plt.title('Box plot for BMI by Smoking Status')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle, Wedge
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

df = pd.read_csv('Covid Data.csv')

def create_age_pyramid():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    bins = [0, 18, 30, 45, 60, 75, 90, 120]
    labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-90', '90+']
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
    
    male_data = df[df['SEX'] == 1]
    female_data = df[df['SEX'] == 2]
    
    male_counts = male_data['AGE_GROUP'].value_counts().reindex(labels, fill_value=0)
    female_counts = female_data['AGE_GROUP'].value_counts().reindex(labels, fill_value=0)
    
    y_pos = np.arange(len(labels))
    
    ax1.barh(y_pos, male_counts, color='#3498db', alpha=0.7, label='Мужчины')
    ax1.barh(y_pos, -female_counts, color='#e74c3c', alpha=0.7, label='Женщины')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Количество случаев')
    ax1.set_title('ВОЗРАСТНАЯ ПИРАМИДА COVID-19\nРаспределение по полу и возрасту', 
                fontsize=14, fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i, (m, f) in enumerate(zip(male_counts, female_counts)):
        ax1.text(m + 50, i, f'{m}', va='center', ha='left', fontweight='bold', fontsize=9)
        ax1.text(-f - 50, i, f'{f}', va='center', ha='right', fontweight='bold', fontsize=9)
    
    status_data = df['CLASIFFICATION_FINAL'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
    
    explode = [0.05] * len(status_data)
    
    wedges, texts = ax2.pie(status_data.values, labels=None, startangle=90, 
                           colors=colors, explode=explode, shadow=True)
    
    ax2.set_title('КЛАССИФИКАЦИЯ СЛУЧАЕВ COVID-19\n(по степени тяжести)', 
                fontsize=14, fontweight='bold', pad=20)
    
    legend_labels = [f'Класс {i+1}: {status_data.values[i]} ({status_data.values[i]/len(df)*100:.1f}%)' 
                    for i in range(len(status_data))]
    ax2.legend(wedges, legend_labels, title="Классификация", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.savefig('covid_age_pyramid.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comorbidity_heatmap():
    comorbidities = ['DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 
                    'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']
    
    comorbidity_matrix = pd.DataFrame(index=comorbidities, columns=comorbidities)
    for i, comorb1 in enumerate(comorbidities):
        for j, comorb2 in enumerate(comorbidities):
            if i == j:
                count_single = df[df[comorb1] == 1].shape[0]
                comorbidity_matrix.loc[comorb1, comorb2] = count_single / len(df) * 100
            else:
                count_both = df[(df[comorb1] == 1) & (df[comorb2] == 1)].shape[0]
                comorbidity_matrix.loc[comorb1, comorb2] = count_both / len(df) * 100
    
    russian_names = {
        'DIABETES': 'Диабет', 'COPD': 'ХОБЛ', 'ASTHMA': 'Астма', 
        'INMSUPR': 'Иммуносупр.', 'HIPERTENSION': 'Гипертония',
        'OTHER_DISEASE': 'Др.забол.', 'CARDIOVASCULAR': 'Сердце-сосуд.',
        'OBESITY': 'Ожирение', 'RENAL_CHRONIC': 'ХПН', 'TOBACCO': 'Курение'
    }
    
    comorbidity_matrix.index = [russian_names.get(col, col) for col in comorbidity_matrix.index]
    comorbidity_matrix.columns = [russian_names.get(col, col) for col in comorbidity_matrix.columns]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.zeros_like(comorbidity_matrix.astype(float))
    np.fill_diagonal(mask, 1)
    
    sns.heatmap(comorbidity_matrix.astype(float), 
                annot=True, 
                fmt='.1f',
                cmap='YlOrRd',
                square=True,
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 9, 'weight': 'bold'},
                linewidths=0.5,
                linecolor='white',
                ax=ax)
    
    ax.set_title('ТЕПЛОВАЯ КАРТА КОМОРБИДНОСТЕЙ\nЧастота сопутствующих заболеваний (%)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('covid_comorbidity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_risk_radar():
    risk_factors = ['DIABETES', 'HIPERTENSION', 'OBESITY', 'CARDIOVASCULAR', 'RENAL_CHRONIC']
    risk_names = ['Диабет', 'Гипертония', 'Ожирение', 'Сердечно-сосудистые', 'ХПН']
    
    risk_percentages = [df[df[factor] == 1].shape[0] / len(df) * 100 for factor in risk_factors]
    
    angles = np.linspace(0, 2*np.pi, len(risk_factors), endpoint=False).tolist()
    risk_percentages += risk_percentages[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, risk_percentages, 'o-', linewidth=2, label='Факторы риска', color='#e74c3c')
    ax.fill(angles, risk_percentages, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(risk_names, fontsize=11)
    ax.set_ylim(0, max(risk_percentages) + 5)
    ax.set_yticks(np.arange(0, max(risk_percentages) + 5, 10))
    ax.grid(True)
    
    ax.set_title('РАДАР ФАКТОРОВ РИСКА COVID-19\nРаспространенность сопутствующих заболеваний', 
                size=16, fontweight='bold', pad=20)
    
    for angle, percentage, label in zip(angles[:-1], risk_percentages[:-1], risk_names):
        ax.text(angle, percentage + 2, f'{percentage:.1f}%', 
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('covid_risk_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_infographic():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis('off')
    
    total_cases = len(df)
    mortality_rate = (df['DATE_DIED'] != '9999-99-99').sum() / total_cases * 100
    icu_cases = (df['ICU'] == 1).sum()
    pneumonia_cases = (df['PNEUMONIA'] == 1).sum()
    avg_age = df['AGE'].mean()
    
    stats_text = f"""
    СВОДНАЯ СТАТИСТИКА COVID-19
    
    Всего случаев: {total_cases:,}
    
    Летальность: {mortality_rate:.1f}%
    
    Требовалась ИВЛ: {icu_cases} случаев
    Пневмония: {pneumonia_cases} случаев
    Средний возраст: {avg_age:.1f} лет
    
    Данные основаны на анализе медицинских записей
    пациентов в период пандемии COVID-19
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=14, 
           verticalalignment='top', linespacing=1.8, fontweight='bold',
           bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.7))
    
    ax.set_title('COVID-19: СТАТИСТИЧЕСКАЯ ИНФОГРАФИКА', 
                fontsize=20, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('covid_summary_infographic.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_age_pyramid()
    create_comorbidity_heatmap()
    create_risk_radar()
    create_summary_infographic()

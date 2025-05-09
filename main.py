import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# import xgboost as xgb
from catboost import CatBoostRegressor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOllama
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor, Tool, create_react_agent
from sklearn.ensemble import RandomForestRegressor
from langchain.chains import LLMMathChain
from langchain.agents import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain
from plasTeX import TeXDocument
from sklearn.preprocessing import LabelEncoder
from IPython.display import Math
from langchain_core.prompts import PromptTemplate
from langchain import hub


st.set_page_config(
    page_title="Rossmann LLM Agent", page_icon="🦜🔗🧠", layout="wide", initial_sidebar_state="collapsed"
)
st.title("🧠 Rossmann 店铺营业额分析")

# # Sidebar API Key
# openai_api_key = st.sidebar.text_input("🔑 输入你的 OpenAI API Key", type="password")
# os.environ["OPENAI_API_KEY"] = openai_api_key

# 上传文件
# trainFile = st.file_uploader("请上传训练数据文件", type="csv", accept_multiple_files=True)
# storeFile = st.file_uploader("请上传附加数据文件", type="csv", accept_multiple_files=True)
# 一次上传多个文件
uploaded_files = st.file_uploader("请上传数据文件", type=["csv", "txt"], accept_multiple_files=True)

monthDict = {
    "一月份": 1,
    "二月份": 2,
    "三月份": 3,
    "四月份": 4,
    "五月份": 5,
    "六月份": 6,
    "七月份": 7,
    "八月份": 8,
    "九月份": 9,
    "十月份": 10,
    "十一月份": 11,
    "十二月份": 12
}

selected = st.selectbox("请选择月份:", list(monthDict.keys()))
month = monthDict[selected]

with st.form(key="form"):
    submit_clicked = st.form_submit_button("确认提交")

if submit_clicked:
    trainFlag, storeFlag, fieldFlag = False, False, False
    for uploaded_file in uploaded_files:
        if "train" in uploaded_file.name:
            trainFlag = True
        if "store" in uploaded_file.name:
            storeFlag = True
        if "field" in uploaded_file.name:
            fieldFlag = True
    if not trainFlag:
        st.error("❌ 缺少包含训练数据的文件，请上传训练数据文件")
    elif not storeFlag:
        st.error("❌ 缺少包含附加数据的文件，请上传附加数据文件")
    elif not fieldFlag:
        st.error("❌ 缺少包含字段解释的文件，请上传相关数据文件")
    else:
        st.success(f"✅ 文件上传成功！将分析{selected}的营业额数据")
        with st.spinner("读取与处理数据中..."):
            for uploaded_file in uploaded_files:
                if "train" in uploaded_file.name:
                    train = pd.read_csv(uploaded_file, parse_dates=['Date'])
                elif "store" in uploaded_file.name:
                    store = pd.read_csv(uploaded_file)
                else:
                    fieldContent = uploaded_file.read().decode("utf-8")  # 解码为文本
                    # with open(uploaded_file, 'r', encoding='utf-8') as file:
                    #     fieldContent = file.read()
                
            # print(train.columns)
            
            # 搜查近3个月的数据
            # train = train[(train['Date'] >= '2015-04-01') & (train['Date'] <= '2015-07-31')]
            train['Month'] = train['Date'].dt.to_period('M')
            uniqueMonths = train['Month'].sort_values(ascending=False).unique()
            # top4 = uniqueMonths[:4]
            # march = str(train['Date'].dt.year.max())+"-"+"03"
            # train = train[train['Month']==march]
            
            marchPeriods = uniqueMonths[uniqueMonths.month == month]
            periodMarch = marchPeriods[0]
            period_array = pd.arrays.PeriodArray([periodMarch.ordinal, (periodMarch-1).ordinal], freq='M')
            train = train[train['Month'].isin(period_array)]
            train = train[train['Open'] == 1]  # 保留当天有营业的数据
            
            monthSales = train.groupby(['Store', 'Month'])['Sales'].sum().unstack() # 计算每家商店每个月的总营业额
            # print(monthSales.head())
            monthSales.columns = monthSales.columns.astype(str)
            marFeb = period_array.strftime('%Y-%m').tolist()
            
            # 计算环比率
            for i in range(len(marFeb)-1):
                monthSales[marFeb[i]] = (monthSales[marFeb[i]] - monthSales[marFeb[i+1]]) / monthSales[marFeb[i+1]]
            # print(monthSales.head())
            months = monthSales[marFeb[:1]].reset_index()
            months.columns = ['Store'] + marFeb[:1]
            # print(months.head())
            
            features = train.groupby(['Store', 'Month']).agg({
                'Promo': 'mean',
                'Customers': 'mean',
                'SchoolHoliday': 'mean',
                'StateHoliday': lambda x: (x != '0').mean()
            }).reset_index()
            # 只保留3月份数据
            features = features[features['Month']==periodMarch]
            features = features.merge(store, on='Store', how='left')

            # print(features.head())
            merged = months.merge(features, on='Store', how='left')

            # for col in ['StoreType', 'Assortment']:
            #     merged[col] = LabelEncoder().fit_transform(merged[col].astype(str))
            
            # # 初始化会话状态
            # if 'ml' not in st.session_state:
            #     st.session_state.ml = False
            # if 'agent' not in st.session_state:
            #     st.session_state.agent = False
    
        st.subheader("📈 特征重要性分析（ML）")
        # if st.button("开始ML分析"):
        st.session_state.ml = True
        for col in ['StoreType', 'Assortment']:
            merged[col] = LabelEncoder().fit_transform(merged[col].astype(str))
        Xlist = ['Promo', 'Customers', 'SchoolHoliday', 'StateHoliday', 'StoreType', 'Assortment', 'Promo2', 'CompetitionDistance']
        y = merged[marFeb[0]]
        
        modelCB = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=4, verbose=0, loss_function='RMSE', od_type='Iter', eval_metric='RMSE', od_wait=50)
        modelRF = RandomForestRegressor()
        modelList = [modelCB, modelRF]
        colors = ["Blues_d", "Reds_d"]
        fig, axes = plt.subplots(1, len(modelList), figsize=(9*len(modelList), 6))
        for i in range(len(modelList)):
            if modelList[i].__class__.__name__ == 'CatBoostRegressor':
                modelList[i].fit(merged[Xlist], y)
            else:
                merged = merged.fillna(0)
                modelList[i].fit(merged[Xlist], y)
            importance = pd.Series(modelList[i].feature_importances_, index=Xlist)
            importance = importance.sort_values(ascending=False)
            
            sns.barplot(
                x=importance.values,
                y=importance.index,
                palette=colors[i],
                ax=axes[i]
            )
            axes[i].set_title(f'{modelList[i].__class__.__name__} Feature Importance', fontsize=12)
            axes[i].tick_params(axis='both', labelsize=8)
            axes[i].set_xlabel('Importance Score', fontsize=9)
            axes[i].set_ylabel('Features', fontsize=9)
            # 添加数值标签
            for j, v in enumerate(importance):
                axes[i].text(v, j, f'{v:.3f}', va='center', fontsize=6)
        
        plt.tight_layout()
        st.pyplot(plt.gcf())

        
        st.subheader("🧠 因素解释分析")
        llm = ChatOllama(model="deepseek-r1:14b")
        search = DuckDuckGoSearchAPIWrapper()
        # memory = ConversationBufferMemory(memory_key="chat_history")
        llm_math_chain = LLMMathChain.from_llm(llm)
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events. You should ask targeted questions",
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math",
            )
        ]

        # template = """Answer the following questions as best you can. You have access to the following tools:

        #     {tools}

        #     请使用以下格式:

        #     问题：你必须回答的输入问题
        #     思考：你应该始终思考该做什么
        #     行为：要采取的行为，应该是 [{tool_names}] 之一或无需使用tool
        #     行为输入：行为的输入
        #     观察：行为的结果……
        #     思考：我现在知道最终答案了
        #     最终答案：原始输入问题的最终答案

        #     开始!

        #     问题: {input}
        #     思考:{agent_scratchpad}
        # """
        
        agent = create_pandas_dataframe_agent(
            llm,
            merged,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True
        )
        
        # agentExecutor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
        
        # react_agent = create_react_agent(llm, agent.tools, template)
        # agentExecutor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True)
        
        
        userInput = """
            你是一位精通数据分析的数据科学家，下面有一个格式为DataFrame的数据集合，包含超过1000家商店的营业额环比率及其他因素字段。
            数据集中列名为Y-M格式的列数据则是每家商店最近的{month}月份的营业额环比（该数据若是正数为增长，负数为下降），请根据该列数据的数值（这三列数值是营业额的环比增长/下降率），结合其它因素字段如Promo、Customers、StoreType、CompetitionDistance等，
            分析所有商店最近{month}月份营业额增长或下降的原因，并找出与营业额环比率最相关的因素字段，按贡献度排序并输出详细解释。
            
            注意：DataFrame的数据已经过预处理
            
            其中，数据的字段解释如下：
            {fieldContent}
        """
        prompt = PromptTemplate.from_template(userInput)
        userInput = prompt.format(month=month, fieldContent=fieldContent)
        
        st.session_state.agent = True
        # # assistant响应容器
        # answer_container = st.chat_message("assistant", avatar="🦜")
        # st_callback = StreamlitCallbackHandler(answer_container)
        # cfg = RunnableConfig()
        # cfg["callbacks"] = [st_callback]
        with st.spinner("Agent分析中..."):
            response = agent.run(userInput)
            # response = agentExecutor.invoke({"input":user_input}, cfg)
            
            # 将分析结果写入markdown文件
            with open("/home/cpss/satoshiyuen/rossmann/analysis_result.md", "w", encoding="utf-8") as f:
                f.write(response)
            st.markdown("### 🤖 分析结果")
            st.write(response)

else:
    st.info("请上传完整数据文件！")
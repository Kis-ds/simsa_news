
# condition
import math
import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import plotly.graph_objects as go
import cufflinks as cf
# tree chart
import plotly.express as px
import seaborn as sns
import time
from streamlit_dynamic_filters import DynamicFilters
import xlwings as xw

import warnings
warnings.filterwarnings('ignore')

#################### >> function #####################
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("필터 추가하기", value=True)


    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("- 조회 조건에 추가할 항목을 선택해주세요.", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"검색하고 싶은 '{column}'을 입력해주세요.",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df


def map_color(value, sorted_values, colors):
    index = sorted_values.index(value)
    return colors[index]


####################################### web page style option #######################################
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"#"collapsed"
)

# margin option
left_mg = 0
right_mg = 20
top_mg = 0
btm_mg = 20

####################################### data preprocessing #######################################

# 여기 pickle로 바꿔서 다시 진행할것
# app = xw.App(visible = False)
# book = xw.Book('./data/upload.xlsx')
# origin = book.sheets(1).used_range.options(pd.DataFrame).value
# origin = origin.reset_index()
# app.kill()
origin = pd.read_excel('./data/upload.xlsx')

df = origin[['date_string','input_list', 'title','link','업체명','그룹명', '대분류','중분류','summary','a_score', 'b_score', 'c_score', 'total_score']]
df['date_string'] = (pd.to_datetime(df['date_string']).dt.strftime('%Y-%m-%d'))
df['date_string'] = df['date_string'].astype(str)
df['total_score'] = df['total_score'].astype(float)


df_plot = df[['date_string', '대분류', '중분류', '업체명', 'title', 'summary', 'link','total_score']].copy()
df_plot.columns = ['날짜', '산업', '분야','기업명', '기사제목', '본문요약', '링크','총점']

TODAY_VAR = df.date_string.max()
COMPANY_CNT = len(df.업체명.unique())
#TREE_LEN = math.ceil(COMPANY_CNT * 0.1)

####################################### web page ui / ux #######################################


################ PART01. side bar → filter ################
dynamic_filters = DynamicFilters(df_plot, filters=['날짜', '산업', '분야','기업명','총점'])
with st.sidebar:
    st.header('조건별 전체 기사 조회하기')
    dynamic_filters.display_filters(gap="large")


################ PART02. main page ################
white_space_1, data_space, white_space_2 = st.columns([0.05, 0.9, 0.05])

################ PART02-01. margin ################
with white_space_1:
    st.empty()

################ PART02-02. main content ################
with data_space:
        ### section 01. head
        st.title(":newspaper:채권 유니버스 모니터링")
        ''
        ''
        st.write(f"다음은 금일 수집한 채권유니버스 소속 기업 관련 뉴스 리스트입니다. (조회일자 기준으로 최근 7일간의 기사를 확인하실 수 있습니다.)\n\n**{TODAY_VAR} 9시 기준**으로 수집된 기사 목록입니다.")
        st.write()
        ''
        ''
        ''

        ### section 02. tree chart

        plot_tree_df = df[df['date_string'] == TODAY_VAR].groupby(['중분류','업체명']).count()['title'].reset_index()
        plot_tree_df = plot_tree_df.sort_values(by='title', ascending=False).reset_index(drop=True)
        st.subheader('오늘의 이슈',
                         help="수집한 결과를 차트로 파악하기 쉽게 트리 차트로 도식화된 화면입니다.\n- 금일 기준 유의미하다고 판단된 기사를 업종(중분류기준)과 기업별로 도식화 하였습니다.\n- 각 업종과 기업별로 클릭하여 각 범위별 비중을 확인할 수 있습니다.\n- 다시 전체 리스트를 조회하고 싶다면 차트 상단의 짙은 남색부분을 클릭해주시면 됩니다. ")
        st.write(f'현재 관리 대상인 전체 391개의 기업 대상으로 수집된 기사 중 유의미하다고 판단된 기사(7일 기준)는 총 {df.shape[0]}건 이며, 금일 수집된 주요 기사는 총 {df[df["date_string"] == TODAY_VAR].shape[0]}건 입니다.')
        st.write(f'최근 주요 기사가 많이 발간된 기업은 **{plot_tree_df["업체명"][0]}**이며, {TODAY_VAR}일 기준 총 **{plot_tree_df["title"][0]}**건이 수집되었습니다.')
        ''
        ''

        st.markdown(
            '<div style = "color:white; font-size: 16px; text-align:center; background-color: #2d3d4a">금일 업종별 주요 뉴스 현황</div>',
            unsafe_allow_html=True)
        st.write()
        fig = px.treemap(plot_tree_df, # plot_tree_df.loc[:TREE_LEN]
                         path = ['중분류','업체명'],
                         values='title',
                         color='title',
                         color_continuous_scale='Sunsetdark')#'RdGn'



        # tree option
        fig.update_traces(textfont=dict(size=16))
        fig.update_traces(root_color="lightgrey")
        fig.update_traces(hovertemplate='<b>중분류</b>: %{label}<br><b>업체명</b>: %{parent}<br><b>집계 뉴스 수 </b>: %{value}')
        fig.update_layout( margin = dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig, use_container_width=True)



        ''
        ''

        ### section 03. sub plot

        st.subheader('주별 이슈 현황',
                         help="수집한 결과를 차트로 파악하기 쉽게 라인그래프와 막대그래프로 도식화한 화면입니다.\n- 산업별 뉴스 현황 : 7일간 전체 산업별(대분류 기준) 기사 발간 현황입니다.\n- 기준별 주요 뉴스 발간 현황 : 각 주요 평가 기준인 산업/ 사업/ 재무 중 기준값 이상으로 주요하다고 판단된 뉴스 동향입니다.")
        company_chart, subject_chart = st.columns(2, gap="large")
        ##### CHAT01. 뉴스 주제별 PIE chat

        with company_chart:
            line_chart_df = df.copy()
            st.markdown(
                '<div style = "color:white; font-size: 16px; text-align:center; background-color: #2d3d4a">산업별 뉴스 현황</div>',
                # #403466  2d3d4a
                unsafe_allow_html=True)

            # text
            check_line = line_chart_df[line_chart_df['date_string'] == TODAY_VAR].groupby(['대분류']).count()[
                'total_score'].reset_index().sort_values(by='total_score', ascending=False).reset_index(drop=True)
            check_line['per'] = round(check_line['total_score'] / check_line['total_score'].sum() * 100, 1)
            ''
            ''
            st.write(f"{TODAY_VAR} 기준 {check_line.loc[0, '대분류']}은 {check_line.loc[0, 'total_score']}건으로 금일 전체 기사 중 {check_line.loc[0, 'per']}%로 제일 많습니다.")

            # multi line chart
            #st.markdown('<h3 style="text-align:center">   </h3>', unsafe_allow_html=True)
            clist = line_chart_df['대분류'].unique().tolist()

            dfs = {cate: line_chart_df[line_chart_df["대분류"] == cate].groupby(['date_string']).count()['total_score'].reset_index() for cate in
                   clist}

            fig = go.Figure()
            for cate, line_chart_df in dfs.items():
                fig = fig.add_trace(go.Scatter(x=line_chart_df["date_string"], y=line_chart_df["total_score"], name=cate))


            fig.update_layout(margin_l=left_mg, margin_r=right_mg, margin_t=top_mg, margin_b=btm_mg,width=400, height=300,
                                  plot_bgcolor='white', paper_bgcolor='white', font_color="black")  # orientation='h'

            st.plotly_chart(fig, use_container_width=True)




            # 원래  pie chart 였음
            # pie_chart_df = df['대분류'].value_counts().reset_index()
            # pie_chart_df.columns = ['산업', '카운트']
            #
            # fig_pie = pie_chart_df.iplot(kind='pie', labels='산업', values='카운트', asFigure=True, dimensions=(400, 350),
            #                              colors=('#a2b4cd', '#a2a9cd', '#77adda', '#0eccfb', '#85d3e6',
            #                                      '#2ebbc9'))  # '#ff4388', '#fe7e22', '#fbc120', '#4b1a84'
            #
            # fig_pie.update_layout(margin_l=left_mg, margin_r=right_mg, margin_t=top_mg, margin_b=btm_mg,
            #                       plot_bgcolor='white', paper_bgcolor='white', font_color="black",
            #                       # plot_bgcolor='#151121', paper_bgcolor='#0e1117',font_color="blue",
            #                       legend=dict(bgcolor='#e7f6fa', orientation='v', font=dict(color='black'))
            #                       # orientation='h' #d9d9d9
            #
            #                       )
            #
            # st.plotly_chart(fig_pie, use_container_width=True)


        # 이거 확인이 필요함
        # a : 산업 / 사업 : b / 재무 : c / 이벤트리스크 : d
        with subject_chart:

            # 집계 기준값
            standard_value = 14
            st.markdown(
                '<div style = "color:white; font-size: 16px; text-align:center; background-color: #2d3d4a">기준별 주요 뉴스 발간 현황</div>',
                # #403466  2d3d4a
                unsafe_allow_html=True)
            st.markdown('<h3 style="text-align:center">   </h3>', unsafe_allow_html=True)
            df_pivot = df.groupby('date_string').apply(lambda x: pd.Series({
                '산업': (x['a_score'] >= standard_value).sum(),
                '사업': (x['b_score'] >= standard_value).sum(),
                '재무': (x['c_score'] >= standard_value).sum().sum()
            }))

            fig_amt = df_pivot.iplot(kind='bar', barmode='stack', asFigure=True, dimensions=(400, 400),
                                     colors=('#7f999f', '#a2a9cd', '#77adda', '#85d3e6', '#0eccfb', '#2ebbc9'))

            fig_amt.update_layout(margin_l=left_mg, margin_r=right_mg, margin_t=top_mg, margin_b=btm_mg,
                                  plot_bgcolor='white', paper_bgcolor='white', font_color="black",
                                  legend=dict(bgcolor='#e7f6fa', yanchor='top', y=-0.1, xanchor='left',
                                              x=0.015, orientation='h', font=dict(color='black'))  # orientation='h'
                                  )

            fig_amt.update_xaxes(showgrid=True, gridcolor='#adb0c2', tickfont_color='black')
            fig_amt.update_yaxes(showgrid=True, gridcolor='#adb0c2', tickfont_color='black')  # 332951
            st.plotly_chart(fig_amt, use_container_width=True)
            # st.write('')


################################### 택1 필요한 부분
        ## 택1-ver01. condition search
        ### section 04. condition search
        # df_plot = df[['date_string','대분류', '중분류','그룹명', '업체명','title','link',  'summary']].copy()
        # df_plot.columns = ['날짜','산업','분야','계열','기업명','기사제목','링크','본문요약']
        # st.dataframe(filter_dataframe(df_plot))
        #st.write(df.dtypes)

        st.subheader('조건별 전체 뉴스 조회하기',
                         help="각 조건별 전체 뉴스 목록을 조회하는 화면입니다. 옆에 있는 창(side bar)를 열어두시면 필요한 목록을 선택할 수 있습니다.")
        dynamic_filters.display_df(height=1500, hide_index=True,
                           column_config={
                               "링크": st.column_config.LinkColumn("URL"),

                               "총점": st.column_config.ProgressColumn(
                                   "총점",
                                   help="산업, 사업, 재무별 평가기준을 통해 부여된 점수의 총합 입니다.",
                                   format="%f점",
                                   min_value=0,
                                   max_value=60,
                                   ),
                               }
                        )

        # from streamlit_pills import pills
        #
        # selected = pills("Label", ["Option 1", "Option 2", "Option 3"], ["🍀", "🎈", "🌈"])
        # st.write(selected)




################ PART03.margin ################
with white_space_2:
    st.empty()

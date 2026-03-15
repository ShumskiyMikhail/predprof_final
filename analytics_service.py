import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo


class AnalyticsService:

    @staticmethod
    def _to_html(fig):
        """Конвертирует Plotly figure в HTML-div без лишних скриптов."""
        return pyo.plot(fig, include_plotlyjs=False, output_type='div')

    @staticmethod
    def _layout(title):
        return dict(
            title=dict(text=title, font=dict(color='#e0e6ed', size=14)),
            paper_bgcolor='#0f1a2a',
            plot_bgcolor='#0a1221',
            font=dict(color='#e0e6ed'),
            margin=dict(l=50, r=20, t=45, b=45),
            xaxis=dict(gridcolor='#1e2d45', zerolinecolor='#1e2d45'),
            yaxis=dict(gridcolor='#1e2d45', zerolinecolor='#1e2d45'),
        )

    def generate_all_plots(self, history, train_y, val_y, test_y, test_preds):
        plots = {}

        fig = go.Figure()
        epochs = list(range(1, len(history['val_acc']) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_acc'],
            mode='lines+markers',
            name='Val Accuracy',
            line=dict(color='#00d1ff', width=2),
            marker=dict(size=7)
        ))
        if 'train_acc' in history:
            fig.add_trace(go.Scatter(
                x=epochs, y=history['train_acc'],
                mode='lines+markers',
                name='Train Accuracy',
                line=dict(color='#ff9900', width=2, dash='dash'),
                marker=dict(size=7)
            ))
        fig.update_layout(
            **self._layout('Точность по эпохам обучения'),
            xaxis_title='Эпоха',
            yaxis_title='Точность',
            legend=dict(bgcolor='#0f1a2a', bordercolor='#1e2d45'),
            dragmode='zoom',
        )
        plots['epochs'] = self._to_html(fig)

        cls, counts = np.unique(train_y, return_counts=True)
        fig = go.Figure(go.Bar(
            x=cls.astype(str), y=counts,
            marker_color='#7000ff',
            text=counts, textposition='outside',
            textfont=dict(color='white', size=11),
        ))
        fig.update_layout(
            **self._layout('Классы в обучающем наборе'),
            xaxis_title='Цивилизация (класс)',
            yaxis_title='Количество записей',
            dragmode='zoom',
        )
        plots['dist'] = self._to_html(fig)

        correct_conf = [float(test_preds[i][test_y[i]]) for i in range(len(test_y))]
        colors = ['#00ff88' if c >= 0.5 else '#ff4444' for c in correct_conf]
        fig = go.Figure(go.Bar(
            x=list(range(len(test_y))),
            y=correct_conf,
            marker_color=colors,
            name='Уверенность',
        ))
        fig.add_hline(y=0.5, line_dash='dash', line_color='white',
                      annotation_text='Порог 0.5', annotation_position='top right')
        fig.update_layout(
            **self._layout('Точность определения каждой записи тестового набора'),
            xaxis_title='Индекс записи',
            yaxis_title='Уверенность сети',
            dragmode='zoom',
        )
        plots['test_detail'] = self._to_html(fig)

        v_cls, v_cnt = np.unique(val_y, return_counts=True)
        top5_idx = np.argsort(v_cnt)[-5:]
        fig = go.Figure(go.Bar(
            x=v_cnt[top5_idx],
            y=v_cls[top5_idx].astype(str),
            orientation='h',
            marker_color='#ff0055',
            text=v_cnt[top5_idx], textposition='outside',
            textfont=dict(color='white', size=11),
        ))
        fig.update_layout(
            **self._layout('Топ-5 классов (Validation)'),
            xaxis_title='Количество записей',
            yaxis_title='Цивилизация (класс)',
            dragmode='zoom',
        )
        plots['top5'] = self._to_html(fig)

        return plots

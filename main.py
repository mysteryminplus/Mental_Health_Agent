import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import base64
from io import BytesIO
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
from enhanced_mental_health_agent import EnhancedMentalHealthAgent

class StreamlitMentalHealthApp:
    def __init__(self):
        self.agent = EnhancedMentalHealthAgent()
        self.setup_streamlit()

    def setup_streamlit(self):
        """Initialize Streamlit interface"""
        st.set_page_config(
            page_title="Mental Health AI Assistant",
            page_icon="🧠",
            layout="wide"
        )

        # Initialize session state variables
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
        if 'interaction_history' not in st.session_state:
            st.session_state.interaction_history = []
        if 'current_stress_level' not in st.session_state:
            st.session_state.current_stress_level = None
        if 'monitoring_start_time' not in st.session_state:  # Add this
            st.session_state.monitoring_start_time = None

    def run(self):
        """Main Streamlit application"""
        st.title("🧠 Mental Health AI Assistant")

        # Sidebar for navigation
        self.sidebar_navigation()

        # Main content based on selected page
        if st.session_state.get('page') == 'voice_analysis':
            self.voice_analysis_page()
        elif st.session_state.get('page') == 'text_analysis':
            self.text_analysis_page()
        elif st.session_state.get('page') == 'emotional_expression':
            self.emotional_expression_page()
        elif st.session_state.get('page') == 'monitoring':
            self.monitoring_page()
        else:
            self.home_page()

    def sidebar_navigation(self):
        """Create sidebar navigation"""
        st.sidebar.title("Navigation")

        pages = {
            'Home': 'home',
            'Voice Analysis': 'voice_analysis',
            'Text Analysis': 'text_analysis',
            'Emotional Expression': 'emotional_expression',
            'Monitoring Dashboard': 'monitoring'
        }

        selection = st.sidebar.radio("Go to", list(pages.keys()))
        st.session_state.page = pages[selection]

        # Monitoring control in sidebar
        st.sidebar.divider()
        self.monitoring_controls()

    def home_page(self):
        """Render home page"""
        st.header("Welcome to Mental Health AI Assistant")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Current Status")
            if st.session_state.current_stress_level:
                self.display_stress_meter(st.session_state.current_stress_level)

            st.subheader("Quick Actions")
            if st.button("Start Quick Analysis"):
                self.quick_analysis()

        with col2:
            st.subheader("Recent Activities")
            self.display_recent_activities()

    def voice_analysis_page(self):
        """Render voice analysis page"""
        st.header("Voice Stress Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Record Voice")
            duration = st.slider("Recording duration (seconds)", 3, 15, 5)
            
            if st.button("Start Recording"):
                self.record_and_analyze_voice(duration)
        
        with col2:
            st.subheader("Analysis Results")
            if st.session_state.get('voice_analysis'):
                self.display_voice_analysis(st.session_state.voice_analysis)


    def text_analysis_page(self):
        """Render text analysis page"""
        st.header("Text Analysis & Support")
        
        # Text input
        user_text = st.text_area("Share how you're feeling...", height=150)
        
        if st.button("Analyze"):
            if user_text:
                with st.spinner("Analyzing your text..."):
                    analysis = self.agent.analyze_context(user_text)
                    self.display_text_analysis(analysis)
                    
                    # Get and display music recommendations
                    recommendations = self.agent.get_personalized_music(
                        st.session_state.current_stress_level
                    )
                    self.display_recommendations(recommendations)

    def emotional_expression_page(self):
        """Render emotional expression assistant page"""
        st.header("Emotional Expression Assistant")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            emotion_type = st.selectbox(
                "What type of emotion do you want to express?",
                ['love', 'gratitude', 'apology']
            )
            
            context_type = st.selectbox(
                "In what context?",
                ['romantic', 'familial', 'friendship', 'professional']
            )
            
            recipient = st.text_input("Who is this message for?")

        with col2:
            specific_details = st.text_area(
                "Add any specific details or memories you'd like to include"
            )
        
        if st.button("Generate Expression"):
            if recipient and specific_details:
                expression = self.generate_emotional_expression(
                    emotion_type,
                    context_type,
                    recipient,
                    specific_details
                )
                st.subheader("Generated Emotional Expression:")
                st.write(expression)

    def monitoring_page(self):
        """Render monitoring dashboard"""
        st.header("Monitoring Dashboard")
        
        # Display monitoring status
        status_col, stats_col = st.columns(2)
        
        with status_col:
            self.display_monitoring_status()
        
        with stats_col:
            self.display_monitoring_stats()
        
        # Historical data visualization
        st.subheader("Stress Level History")
        self.plot_stress_history()

    def record_and_analyze_voice(self, duration):
        """Record and analyze voice"""  # Docstring is fine, but the code must be indented
        try:
            with st.spinner(f"Recording for {duration} seconds..."):
                analysis = self.agent.analyze_voice(duration)

                if analysis:  # Check if analysis is not None
                    st.session_state.voice_analysis = analysis
                    st.session_state.current_stress_level = analysis['stress_level']
                    self.update_history('voice_analysis', analysis)

                    # Display results (including emotion)
                    st.write("Emotion:", analysis['emotion'])  # Display the emotion
                    self.display_stress_meter(analysis['stress_level'])
                    # ... (display other results) ...

                    st.success("Recording and analysis complete!")
                else:
                    st.warning("Voice analysis failed. Please try again.")

        except Exception as e:
            st.error(f"Error during voice analysis: {str(e)}")


    def display_voice_analysis(self, analysis):
        """Display voice analysis results"""
        st.write("Stress Level:", analysis['stress_level'])
        st.write("Confidence:", f"{analysis['confidence']*100:.1f}%")
        
        # Display voice features
        features = analysis['voice_features']
        fig = go.Figure(data=[
            go.Bar(
                x=list(features.keys()),
                y=[float(v) for v in features.values()]
            )
        ])
        fig.update_layout(title="Voice Features Analysis")
        st.plotly_chart(fig)

        if analysis['emotion'] in ['sad', 'neutral']:
            recommended_music = self.get_personalized_music(analysis['emotion'])
            if recommended_music:
                st.write("Recommended Song:", recommended_music['title'], "by", recommended_music.get('artist', 'Unknown'))
                st.audio('temp_song.mp3', format='audio/mp3')

    def display_text_analysis(self, analysis):
        """Display text analysis results"""
        # Emotions
        emotions_df = pd.DataFrame(analysis['emotions'])
        fig = px.bar(
            emotions_df,
            x='label',
            y='score',
            title="Detected Emotions"
        )
        st.plotly_chart(fig)
        
        # Key phrases and entities
        col1, col2 = st.columns(2)
        with col1:
            st.write("Key Phrases Detected:")
            for phrase in analysis['key_phrases']:
                st.write(f"• {phrase}")
        
        with col2:
            st.write("Entities Detected:")
            for entity, label in analysis['entities']:
                st.write(f"• {entity} ({label})")

        if analysis.get('suggestion'):
            st.subheader("Support Message:")
            st.write(analysis['suggestion'])

    def display_recommendations(self, recommendations):
        """Display music recommendations"""
        if recommendations:
            st.subheader("Music Recommendations")
            for rec in recommendations:
                with st.expander(f"🎵 {rec['title']}"):
                    st.write(f"Artist: {rec.get('artist', 'Unknown')}")
                    st.write(f"Type: {rec['type']}")
                    st.write(f"[Listen Here]({rec['link']})")

    def display_stress_meter(self, stress_level):
        """Display stress level meter"""
        stress_values = {
            'low_stress': 0.3,
            'moderate_stress': 0.6,
            'high_stress': 0.9
        }
        
        value = stress_values.get(stress_level, 0.5)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value * 100,
            title = {'text': "Stress Level"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}
                ]
            }
        ))
        
        st.plotly_chart(fig)

    def monitoring_controls(self):
        """Monitoring controls in sidebar"""
        st.sidebar.subheader("Monitoring Control")

        if st.session_state.monitoring_active:
            if st.sidebar.button("Stop Monitoring"):
                self.agent.stop_monitoring()
                st.session_state.monitoring_active = False
                st.session_state.monitoring_start_time = None  # Reset start time
        else:
            if st.sidebar.button("Start Monitoring"):
                self.agent.start_monitoring()
                st.session_state.monitoring_active = True
                st.session_state.monitoring_start_time = datetime.now()

    def quick_analysis(self):
        """Perform quick analysis"""
        with st.spinner("Performing quick analysis..."):
            # Voice analysis
            voice_analysis = self.agent.analyze_voice(5)

            # Check if voice_analysis is None before accessing it
            if voice_analysis is not None:  # This is the crucial check
                st.session_state.current_stress_level = voice_analysis['stress_level']
                self.update_history('quick_analysis', voice_analysis)

                # Display results
                self.display_stress_meter(voice_analysis['stress_level'])

                # Show recommendations
                recommendations = self.agent.get_personalized_music(
                    voice_analysis['stress_level']
                )
                self.display_recommendations(recommendations)
            else:
                st.warning("Voice analysis failed. Please try again.")  # Display a message to the user

    def update_history(self, interaction_type, data):
        """Update interaction history with stress level"""
        try:
            # Your code to update the history goes here.  For example:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current time
            conn = sqlite3.connect('mental_health_agent.db', check_same_thread=False)
            cursor = conn.cursor()

            # Example: Insert data into a table (replace with your table and columns)
            cursor.execute("INSERT INTO user_interactions (timestamp, type, data) VALUES (?, ?, ?)", (timestamp, interaction_type, str(data)))
            conn.commit()
            conn.close()

        except Exception as e:
            st.error(f"Error updating history: {e}") # Handle error appropriately


    def plot_stress_level_history(self):
        """Plot stress level history"""
        try:
            # Your code to plot the stress level history goes here.
            # Example using Plotly:
            conn = sqlite3.connect('mental_health_agent.db', check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, stress_level FROM user_interactions WHERE type='voice_analysis'") # Example query, adjust as needed.
            data = cursor.fetchall()
            conn.close()

            if data: # Check if any data was retrieved
                timestamps = [row[0] for row in data]
                stress_levels = [row[1] for row in data]

                fig = go.Figure(data=[go.Scatter(x=timestamps, y=stress_levels, mode='lines+markers')])
                st.plotly_chart(fig)
            else:
                st.write("No stress level history available yet.") # Handle no data case

        except Exception as e:
            st.error(f"Error plotting stress level history: {e}")  # Handle errors


    def generate_emotional_expression(self, emotion_type, context_type, recipient, specific_details):
        """Generates an emotional expression."""
        try:
            # Your code to generate the emotional expression goes here.
            # Example (replace with your logic):
            if emotion_type == "sadness":
                expression = f"Dear {recipient}, I understand you're feeling sad.  {specific_details}.  Remember, brighter days are ahead."
            elif emotion_type == "joy":
                expression = f"Dear {recipient}, I'm so happy to hear about your joy! {specific_details}. Keep shining!"
            else:
                expression = f"Dear {recipient}, I'm here for you. {specific_details}."  # Default

            return expression  # Return the generated expression

        except Exception as e:
            st.error(f"Error generating expression: {e}") # Handle the error appropriately
            return "" # Or return a default message

          # Display the generated text



    def display_monitoring_status(self):
        """Display monitoring status"""
        st.metric(
            "Monitoring Status",
            "Active" if st.session_state.monitoring_active else "Inactive"
        )

        if st.session_state.monitoring_active:
            st.write("Monitoring since:", st.session_state.monitoring_start_time)
            
    def display_monitoring_stats(self):
        """Display monitoring statistics"""
        if st.session_state.interaction_history:
            total_interactions = len(st.session_state.interaction_history)
            avg_stress = np.mean([
                interaction['data'].get('stress_level', 0)
                for interaction in st.session_state.interaction_history
            ])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Interactions", total_interactions)
            with col2:
                st.metric("Average Stress Level", f"{avg_stress:.2f}")
    
    def display_recent_activities(self):
        """Displays recent user activities from the database."""
        try:
            conn = sqlite3.connect('mental_health_agent.db', check_same_thread=False)
            cursor = conn.cursor()

            # Retrieve recent activities (adjust the query as needed)
            cursor.execute("SELECT * FROM user_interactions ORDER BY timestamp DESC LIMIT 10")  # Example query
            activities = cursor.fetchall()
            conn.close()

            if activities:
                # Convert to Pandas DataFrame for easier display
                df = pd.DataFrame(activities, columns=['Timestamp', 'Stress Level', 'Emotion', 'Text Content', 'Intervention Provided'])  # Replace column names
                st.dataframe(df)  # Display the DataFrame in Streamlit
            else:
                st.write("No recent activities found.")

        except sqlite3.Error as e:
            st.error(f"Error retrieving activities: {e}")


if __name__ == "__main__":
    app = StreamlitMentalHealthApp()
    app.run()
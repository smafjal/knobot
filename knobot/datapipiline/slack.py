class SlackData:
    def __init__(self, channel_id: str, message_id: str, message: str):
        self.channel_id = channel_id
        self.message_id = message_id
        self.message = message

    def get_message(self):
        return self.message

    def get_channel_id(self):
        return self.channel_id
    
    def get_message_id(self):
        return self.message_id

class SlackDataPipeline:
    def __init__(self, slack_data: SlackData):
        if slack_data is None:
            self.slack_data = self.demoData()
        else:
            self.slack_data = slack_data

    def process_data(self):
        return self.slack_data

    def save_data(self):
        pass
    
    def load_data(self):
        pass

    def demoData(self):
        data = [
            {
                "instruction": "How do I reset my Slack password?",
                "response": "Go to your Slack profile settings, then select 'Password & Authentication' to reset your password."
            },
            {
                "instruction": "What is the procedure for submitting a Confluence page for review?",
                "response": "Once your Confluence page is ready, click the 'Submit for review' button and assign it to the relevant team member."
            },
            {
                "instruction": "Who do I contact for IT support?",
                "response": "You can reach out to the IT support team via Slack channel #it-support or email it-support@company.com."
            }
        ]
        return data
# Ma trận Thái độ (5 Cảm xúc x 14 Cử chỉ) - Từ vựng Học thuật & Lọc mâu thuẫn
ATTITUDE_MATRIX = {
    "angry": {
        "call": "Demand",
        "palm": "Reject",
        "stop": "Halt",
        "hand_heart": "Undefined", 
        "fist": "Threaten",
        "middle_finger": "Insult",
        "ok": "Undefined",         
        "peace": "Undefined",      
        "point": "Accuse",
        "one": "Warn",
        "holy": "Undefined",       
        "rock": "Rage",
        "dislike": "Despise",
        "like": "Undefined"        
    },
    "happy": {
        "call": "Invite",
        "palm": "Greet",
        "stop": "Undefined",       
        "hand_heart": "Love",
        "fist": "Celebrate",
        "middle_finger": "Undefined", 
        "ok": "Approve",
        "peace": "Chill",
        "point": "Choose",
        "one": "Best",
        "holy": "Bless",
        "rock": "Party",
        "dislike": "Undefined",    
        "like": "Praise"
    },
    "neutral": {
        "call": "Contact",
        "palm": "Wait",
        "stop": "Pause",
        "hand_heart": "Care",
        "fist": "Ready",
        "middle_finger": "Disrespect",
        "ok": "Accept",
        "peace": "Hello",
        "point": "Show",
        "one": "Single",
        "holy": "Pray",
        "rock": "Cool",
        "dislike": "Disagree",
        "like": "Agree"
    },
    "sad": {
        "call": "Beg",
        "palm": "Hopeless",
        "stop": "Surrender",
        "hand_heart": "Heartbreak",
        "fist": "Endure",
        "middle_finger": "Undefined", 
        "ok": "Cope",
        "peace": "Farewell",
        "point": "Regret",
        "one": "Lonely",
        "holy": "Mercy",
        "rock": "Undefined",       
        "dislike": "Disappoint",
        "like": "Undefined"        
    },
    "surprise": {
        "call": "Startle",
        "palm": "Shock",
        "stop": "Hesitate",
        "hand_heart": "Undefined", 
        "fist": "Gasp",
        "middle_finger": "Undefined", 
        "ok": "Amazed",
        "peace": "Stunned",
        "point": "Discover",
        "one": "Unbelievable",
        "holy": "Miracle",
        "rock": "Hype",
        "dislike": "Undefined",    
        "like": "Impress"
    }
}

def get_attitude(emotion, gesture):
    if emotion in ["None", "Đang chờ..."] or gesture in ["None", "Đang chờ..."]:
        return "Waiting..."
    
    e = emotion.lower()
    g = gesture.lower()
    
    if e in ATTITUDE_MATRIX and g in ATTITUDE_MATRIX[e]:
        return ATTITUDE_MATRIX[e][g]
        
    return "Undefined"
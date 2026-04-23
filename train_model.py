import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data = {
    "text":[
        "SSC CGL 2024 official notification released on ssc.nic.in",
        "UPSC civil services prelims admit card download at upsc.gov.in",
        "Indian Railway recruitment RRB NTPC apply on rrbcdg.gov.in",
        "SBI PO recruitment 2024 apply on sbi.co.in careers page",
        "IBPS clerk notification official portal ibps.in",
        "NDA exam form available at joinindianarmy.nic.in",
        "HAL recruitment for apprentice posts official notification",
        "BPSC 70th combined exam notification at bpsc.bih.nic.in",
        "DRDO scientists recruitment CEPTAM notification 2024",
        "Post office GDS recruitment apply at indiapostgdsonline.gov.in",
        "TNPSC group 2 exam schedule released on tnpsc.gov.in",
        "ESIC staff nurse recruitment official vacancy announced",
        "Delhi police constable recruitment apply on ssc.nic.in",
        "Coal India management trainee recruitment official 2024",
        "AIIMS NORCET nursing officer exam notification released",
        "MP police constable bharti official notification mppolice.gov.in",
        "RPSC RAS prelims 2024 application begins on rpsc.rajasthan.gov.in",
        "ONGC graduate trainee recruitment official notification engineering",
        "FCI assistant grade 3 exam notification fci.gov.in",
        "Indian Navy SSR AA recruitment apply joinindiannavy.gov.in",
        "UP police SI recruitment 2024 official uppbpb.gov.in apply now",
        "NHM UP staff nurse community health officer vacancy notification",
        "SAIL management trainee MT 2024 notification sailcareers.com",
        "RBI assistant 2024 notification at rbi.org.in",
        "CAPF AC recruitment UPSC notification 2024",
        "NIACL assistant recruitment notification 2024 newindia.co.in",
        "UKPSC lower subordinate services notification official",
        "PGCIL diploma trainee recruitment powergridindia.com",
        "AAI junior executive recruitment aai.aero official portal",
        "BSF head constable ministerial recruitment 2024 bsf.gov.in",
        "Pay Rs 2000 registration fee get guaranteed govt job 2024",
        "Urgent hiring 500 railway vacancies WhatsApp 9999XXXXXX now",
        "Army job guaranteed send Aadhaar and pay processing charges",
        "Work from home govt job 50000 monthly no experience needed",
        "Ministry of Railways direct recruitment no exam pay 1500 fee",
        "Secret UPSC shortcut get IAS in 30 days pay now limited seats",
        "Police bharti guaranteed selection pay 5000 advance",
        "Govt job offer letter ready pay security deposit to release",
        "PMO office urgent hiring apply via Telegram link only",
        "Bank job confirmed OBC quota vacancy pay 3000 form fee",
        "10th pass railway job no exam required pay joining fee",
        "SSC CGL 2024 result leaked buy answer key contact us",
        "Free coaching and guaranteed govt job pay Rs 8000 token amount",
        "Defence ministry contractor hiring pay refundable deposit",
        "Earn 60000 monthly data entry work from home govt scheme",
        "ISRO vacancy for 12th pass no exam direct joining pay Rs 1000",
        "High court peon job guaranteed pay money get appointment letter",
        "NDA selection without exam contact agent pay 10000 now",
        "Immediate joining central government job WhatsApp link click here",
        "LIC HFL job offer send personal details and pay processing fee",
        "1200 vacancies Indian Post direct joining no exam Telegram group",
        "Fake govt portal applygovernmentjobs.in fill form pay Rs 500",
        "District collector office peon vacancy no exam pay advance",
        "100 percent job guarantee IPS IAS coaching pay enrolment fee urgent",
        "CBI officer recruitment contact number WhatsApp 8000 fee required",
        "Railway group D confirmed vacancy pay booking fee agent contact",
        "BSNL JTO vacancy 2024 pay Rs 2500 form processing fee apply now",
        "Constable job Delhi police confirmed apply through agent pay now",
        "Income tax department bharti no written test pay security money",
        "Sarkari naukri guaranteed 30000 salary pay 4000 advance fee apply",
        "Government of India Railway recruitment apply now",
        "Ministry job application fee 5000 urgent hiring",
        "UPSC official notification civil service exam",
        "Pay money and get railway job immediately",
        "Indian army recruitment official portal",
        "Govt job guaranteed pay processing fee"
    ],
    "label":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])

y = df["label"]

model = LogisticRegression()

model.fit(X,y)

pickle.dump(model,open("job_model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("Model trained")
import pandas as pd
import numpy as np
import difflib
from difflib import SequenceMatcher
import openai
import time

### AI REFERRALS ###

import pandas as pd
import openai
import time

df = pd.read_csv('/Users/stjames/Dropbox/Pablo/data/legal-clinic-processing/new_report/report.csv',encoding='latin-1',parse_dates=['Intake Date','DOB','Immigration Arrest Date','Attorney Consult Date'], dtype=str)

df = df.drop(df.columns[0], axis=1)

series_list = [df.iloc[i].transpose().rename('value') for i in range(len(df))]

series_string_list = []
for series in series_list:
    name = series[7] + " " + series[6] # new line to compute name
    series_string = ', '.join(f'{index}: {value}' for index, value in series.items())
    series_string = 'Name: ' + name + ', ' + series_string # include name in series_string
    series_string_list.append(series_string)

openai.api_key = ''

sysrole = "You are a highly intelligent AI attorney assistant, tasked with writing concise referrals for detainees seeking pro bono legal aid."

responses = []

prompt_info = """Focus on succinctly presenting the most pertinent information, specifically emphasizing the client's key demographics, their fear of return, criminal convictions, and potential avenues of relief, including bond eligibility. Do not repeat redundant information. Do not make any new information and do not perform legal analysis yourself. 

AI should accurately depict each case, incorporating details such as case status, assigned Immigration Judge (IJ), relief options, case location transfers, prior deportation orders.


Each referral needs to include:
Client's name abbreviated as FirstName-Initials of LastNames.
Key demographics, including language spoken, and nationality.
U.S. County of residence, in the format: resident of XYZ county
A brief outline of criminal convictions, using specific penal codes where possible.
A precise summary of their fear of returning to their native country.
Paths to relief specifically identified by the attorney, in the format - Relief: XYZ
Bond eligibility in the format - Bond: Yes/No
Do not even mention CDSS funding eligibility *unless explicitly stated*
Each referral needs to include the name of the immigration judge (IJ) in the format - IJ: XYZ


AI should omit unconfirmed or superfluous details such as speculative information on ongoing applications, unnecessary client background descriptors, redundant criminal history, overgeneralized relief options, focusing on the most pertinent case details.

Do not include extraneous details such as:
A number, date of birth, or date they were detained by ICE. Do not include Race/ethnicity EXCEPTION: if they are black or indigenous. Do not include Dorm details or pro se packet information / requirements. Do not include extraneous details such as: right to hire an attorney, request for continuance, intention to self-represent, interest in probono representation, COVID-19 vaccination status, absence of mental health consitions, personal and family background details not directly relevant to their legal posture, advice or next steps provided during the legal consultation, location of criminal convictions. Do not include how someone was convicted, such as whether they toook a plea deal or were convicted in trial. Do not even mention CDSS funding eligibility *unless if explicitly stated*


The referrals should be concise, between 50 to 250 words, with a preference for brevity. They should communicate crucial information for potential pro bono representation, and do not feel the need to use complete sentences. Replace the term Latinx with Hispanic, and abbreviate "In-Person Clinic" as IPC, and "Lawful Permanent Resident" as LPR. Stick to the facts and avoid redundancy.
		
Copy the style of the sample referrals pasted below: 

Lara D-M  is an LPR resident of Mendocino County who appears CDSS funding eligible. He seeks rep in a readjustment with 212(h) waiver and/or PCR. He'd be held to the heightened hardship standard on his Negrete readjustment (has a 273.5 DV conviction). His USC wife has a tumor on her throat, is on chemo and can't eat solids, and his USC son is struggling in dad's absence and mom's illness. Other family members have serious health issues but it would require some creative bootstrapping b/c they aren't technically qualifying relatives. He was picked up at fire camp, so has good rehab evidence.

Trevor B-B at GSA is a Spanish-speaking resident of Napa County, citizen of Mexico. He appears CDSS funding eligible and seeks rep in withholding only proceedings, after IJ Riley vacated his negative RFI. Freddy was sexually abused by his brother as a child, and suffers PTSD. He was forced to work for the Sinaloa Cartel from approximately 2007 until 2012. He was held in a house under threat, and forced to cross the border with drugs countless times. He appears to be eligible for a T Visa. In 2012 or 13, he crossed the border with drugs under threat and then fled into the U.S. Freddy may also be U visa eligible as he was slashed with a knife in several places in Santa Barbara County, where he used to be a resident. Freddy has a number of misdemeanor cases in Napa County. He was detained leaving the county jail.

Jaime Carlos C at GSA is a spanish-speaking citizen of Mexico, resident of Napa County, where he was picked up leaving jail. He appears CDSS funding eligible. He appears asylum/WH/CAT eligible and fears return because he has family involved in a cartel and several family members have been killed.

Alfaro R-D- at GSA is a Spanish speaking resident of Arizona, citizen of Mexico. He might be CDSS funding eligible. He passed his RFI and is in withholding-only proceedings. He seeks rep in WH and CAT. He suffered past torture. His father died due to medical negligence, his mom won a major lawsuit, and as a result kidnappers in Mexico thought he had money and kidnapped him and tortured him for 3 days. He fears future harm too.

Alfredo A-B- at GSA is a Spanish-speaking citizen of Honduras, resident of Santa Barbara County. He was persecuted and recruited by gangs in Honduras, and eventually forced to join MS. He was shot at when he tried to flee the gang and eventually had to flee the country; he entered the U.S. as a UAC. He is WH and CAT eligible


Jorge TO at GSA is an English-speaking citizen of Honduras, resident of Kern County. He has been in the US since 1 year old, has 3 USC children, his sister is a Bakersfield police officer, and fears return because he has tattoos and dropped out of the Surenos in prison and went to SNY. He and his mother were also the victims of severe abuse by his father, who he still fears (his father strangled him until he turned blue as an infant). He has only one conviction, a 2014 carjacking. He is diagnosed bipolar but stable on medications. His family spent a lot of money before the IJ and BIA on private representation Andy's not happy with. He has a pending PFR at the 9th Cir. where he's been appointed counsel for mediation only, because he's victim of ICE's data leak. DHS also filed a motion to reopen that's pending at the BIA. Andy seeks rep. in 240 proceedings for WH/CAT. His family could also consider private counsel w/ a payment plan. Andy was a hunger striker.

Alex CD at GSA is an English-speaking former DACA recipient, resident of Stanislaus County. He seeks assistance filing a U Visa based on domestic violence he suffered by his former partner, who also took their child away from him. He filed a police report.

Write the referral for the legal consult provided. Note that it is in a spreadsheet format, and some fields have blank values.""" #Rest of the prompt

for series_string in series_string_list:
    series_responses = []
    for i in range(2):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"system","content":sysrole},{"role":"user","content":prompt_info + series_string}],
            temperature=.1,
            max_tokens=300,
            top_p=.5,
            frequency_penalty=.7,
            presence_penalty=.7
        )
        model_message = response['choices'][0]['message']['content']
        series_responses.append(model_message)
        time.sleep(7)
        
    combined_prompt = """Combine the best parts of the following two responses into a concise, well-structured and coherent response. The response should be concise, between 50 to 250 words, with a preference for brevity. Do not feel the need to use complete sentences. Do not repeat redundant information. Do not make any new information and do not perform legal analysis yourself. 

AI should accurately depict each case, incorporating details such as case status, assigned Immigration Judge (IJ), relief options, case location transfers, prior deportation orders.


Each referral needs to include:
Client's name abbreviated as FirstName-Initials of LastNames.
Key demographics, including language spoken, and nationality.
U.S. County of residence, in the format: resident of XYZ county
A brief outline of criminal convictions, using specific penal codes where possible.
A precise summary of their fear of returning to their native country.
Paths to relief specifically identified by the attorney, in the format - Relief: XYZ
Bond eligibility in the format - Bond: Yes/No
Do not even mention CDSS funding eligibility *unless explicitly stated*
Each referral needs to include the name of the immigration judge (IJ) in the format - IJ: XYZ


AI should omit unconfirmed or superfluous details such as speculative information on ongoing applications, unnecessary client background descriptors, redundant criminal history, overgeneralized relief options, focusing on the most pertinent case details.

Do not include extraneous details such as:
A number, date of birth, or date they were detained by ICE. Do not include Race/ethnicity EXCEPTION: if they are black or indigenous. Do not include Dorm details or pro se packet information / requirements. Do not include extraneous details such as: right to hire an attorney, request for continuance, intention to self-represent, interest in probono representation, COVID-19 vaccination status, absence of mental health consitions, personal and family background details not directly relevant to their legal posture, advice or next steps provided during the legal consultation, location of criminal convictions. Do not include how someone was convicted, such as whether they toook a plea deal or were convicted in trial. Do not even mention CDSS funding eligibility *unless if explicitly stated*


The referrals should be concise, between 50 to 250 words, with a preference for brevity. They should communicate crucial information for potential pro bono representation, and do not feel the need to use complete sentences. Replace the term Latinx with Hispanic, and abbreviate "In-Person Clinic" as IPC, and "Lawful Permanent Resident" as LPR. Stick to the facts and avoid redundancy.
		
Copy the style of the sample referrals pasted below: 

Lara D-M  is an LPR resident of Mendocino County who appears CDSS funding eligible. He seeks rep in a readjustment with 212(h) waiver and/or PCR. He'd be held to the heightened hardship standard on his Negrete readjustment (has a 273.5 DV conviction). His USC wife has a tumor on her throat, is on chemo and can't eat solids, and his USC son is struggling in dad's absence and mom's illness. Other family members have serious health issues but it would require some creative bootstrapping b/c they aren't technically qualifying relatives. He was picked up at fire camp, so has good rehab evidence.

Trevor B-B at GSA is a Spanish-speaking resident of Napa County, citizen of Mexico. He appears CDSS funding eligible and seeks rep in withholding only proceedings, after IJ Riley vacated his negative RFI. Freddy was sexually abused by his brother as a child, and suffers PTSD. He was forced to work for the Sinaloa Cartel from approximately 2007 until 2012. He was held in a house under threat, and forced to cross the border with drugs countless times. He appears to be eligible for a T Visa. In 2012 or 13, he crossed the border with drugs under threat and then fled into the U.S. Freddy may also be U visa eligible as he was slashed with a knife in several places in Santa Barbara County, where he used to be a resident. Freddy has a number of misdemeanor cases in Napa County. He was detained leaving the county jail.

Jaime Carlos C at GSA is a spanish-speaking citizen of Mexico, resident of Napa County, where he was picked up leaving jail. He appears CDSS funding eligible. He appears asylum/WH/CAT eligible and fears return because he has family involved in a cartel and several family members have been killed.

Alfaro R-D- at GSA is a Spanish speaking resident of Arizona, citizen of Mexico. He might be CDSS funding eligible. He passed his RFI and is in withholding-only proceedings. He seeks rep in WH and CAT. He suffered past torture. His father died due to medical negligence, his mom won a major lawsuit, and as a result kidnappers in Mexico thought he had money and kidnapped him and tortured him for 3 days. He fears future harm too.

Alfredo A-B- at GSA is a Spanish-speaking citizen of Honduras, resident of Santa Barbara County. He was persecuted and recruited by gangs in Honduras, and eventually forced to join MS. He was shot at when he tried to flee the gang and eventually had to flee the country; he entered the U.S. as a UAC. He is WH and CAT eligible


Jorge TO at GSA is an English-speaking citizen of Honduras, resident of Kern County. He has been in the US since 1 year old, has 3 USC children, his sister is a Bakersfield police officer, and fears return because he has tattoos and dropped out of the Surenos in prison and went to SNY. He and his mother were also the victims of severe abuse by his father, who he still fears (his father strangled him until he turned blue as an infant). He has only one conviction, a 2014 carjacking. He is diagnosed bipolar but stable on medications. His family spent a lot of money before the IJ and BIA on private representation Andy's not happy with. He has a pending PFR at the 9th Cir. where he's been appointed counsel for mediation only, because he's victim of ICE's data leak. DHS also filed a motion to reopen that's pending at the BIA. Andy seeks rep. in 240 proceedings for WH/CAT. His family could also consider private counsel w/ a payment plan. Andy was a hunger striker.

Alex CD at GSA is an English-speaking former DACA recipient, resident of Stanislaus County. He seeks assistance filing a U Visa based on domestic violence he suffered by his former partner, who also took their child away from him. He filed a police report.

Write the referral for the legal consult provided.: """ + series_responses[0] + " " + series_responses[1]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"system","content":sysrole},{"role":"user","content":combined_prompt}],
        temperature=.1,
        max_tokens=300,
        top_p=.5,
        frequency_penalty=.7,
        presence_penalty=.7
    )
    best_response = response['choices'][0]['message']['content']
    responses.append((series_string, series_responses[0], series_responses[1], best_response))
    time.sleep(7)

with open('/Users/stjames/Dropbox/Pablo/data/legal-clinic-processing/ai_referrals.txt', 'w') as f:
    for response_group in responses:
        start = response_group[0].find("Name: ") + len("Name: ")
        end = response_group[0].find(", Intake Date:")
        name = response_group[0][start:end]
        f.write(name)
        f.write('\n')
        f.write('Combined Response:\n')
        f.write(response_group[3])
        f.write('\n')
        f.write('Response 1:\n')
        f.write(response_group[1])
        f.write('\n')
        f.write('Response 2:\n')
        f.write(response_group[2])
        f.write('\n\n')

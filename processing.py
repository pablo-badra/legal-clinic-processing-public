import pandas as pd
import numpy as np
import difflib
from difflib import SequenceMatcher
import openai
import time

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

### DATA LOADING ###

intl = pd.read_csv('/Users/stjames/Dropbox/Pablo/data/legal-clinic-processing/initial_report/initial_report.csv',parse_dates=['Client Since','Birthdate','Detention Date'], dtype={'A Number':str,'Mailing Zip/Postal Code':str})

# Import new dataframe
new_df = pd.read_csv('/Users/stjames/Dropbox/Pablo/data/legal-clinic-processing/new_report/report.csv', dtype={'AlienNumber': str,'Zip':str}, parse_dates=['Intake Date','DOB','Immigration Arrest Date'], encoding='latin-1')

# Select columns that match the structure of the old dataframe and rename them
# Select columns that match the structure of the old dataframe and rename them
new_df = new_df[['Intake Date','Last Names','First Names','AlienNumber','DOB','Citizenship','Gender','LGBTQ','Ethnicity','City','County','State','Zip','Language','Immigration Arrest Date', 'Immigration Arrest Location','detained under statute','Immigration Arrest Circumstances','Immigration Arrest Notes','Current Immigration Status','ContactPhoneOrEmailRow1','ContactNameRow1','Marital Status']]

new_df.rename(columns={
    'Last Names': 'Last Name',
    'First Names': 'First Name',
    'Language': 'Language',
    'AlienNumber': 'A Number',
    'County': 'Mailing County',
    'DOB': 'Birthdate',
    'Gender': 'Gender Identity/Expression',
    'LGBTQ': 'Sexual Orientation',
    'Ethnicity': 'Ethnicity',
    'City': 'Mailing City',
    'State': 'Mailing State/Province',
    'Zip': 'Mailing Zip/Postal Code',
    'Current Immigration Status': 'Immigration Status',
    'ContactPhoneOrEmailRow1': 'Emergency Contact Phone',
    'ContactNameRow1': 'Emergency Contact Name',
    'Marital Status': 'Marital Status',
    'Citizenship': 'Country of Birth',
    'Immigration Arrest Date': 'Detention Date',
    'Immigration Arrest Location': 'detained_where',
    'detained under statute':'Detention Statute',
    'Immigration Arrest Circumstances': 'detained_how',
    'Immigration Arrest Notes': 'detained_notes'
}, inplace=True)

# Consolidate detention notes
new_df['detained_notes'].fillna('', inplace=True)
new_df['detained_how'].fillna('', inplace=True)
new_df['detained_where'].fillna('', inplace=True)

new_df['Immigration Arrest Circumstances'] = new_df['detained_how'] + ' ' + new_df['detained_where'] + ' Notes: ' + new_df['detained_notes']
new_df.drop(['detained_how','detained_where','detained_notes'], axis=1, inplace=True)

intl.set_index('A Number', inplace=True)
new_df.set_index('A Number', inplace=True)
combined_df = intl.combine_first(new_df)

combined_df['Birthdate'] = pd.to_datetime(combined_df['Birthdate'])
combined_df['Detention Date'] = pd.to_datetime(combined_df['Detention Date'])
combined_df['Client Since'] = pd.to_datetime(combined_df['Client Since'])

cf = combined_df

ice_cob = ['--None--',
 'AFGHANISTAN',
 'ALBANIA',
 'Algeria',
 'American Samoa',
 'Andorra',
 'ANGOLA',
 'Anguilla',
 'Antigua-Barbuda',
 'Arabian Peninsula',
 'ARGENTINA',
 'ARMENIA',
 'Aruba',
 'AUSTRALIA',
 'Austria',
 'Azerbaijan',
 'BAHAMAS',
 'Bahrain',
 'BANGLADESH',
 'Barbados',
 'Belarus',
 'Belgium',
 'BELIZE',
 'Benin',
 'Bermuda',
 'Bhutan',
 'BOLIVIA',
 'Born On Board Ship',
 'Bosnia-Herzegovina',
 'Botswana',
 'BRAZIL',
 'British Virgin Islands',
 'Brunei',
 'BULGARIA',
 'BURKINA FASO',
 'Burma',
 'Burundi',
 'CAMBODIA',
 'CAMEROON',
 'CANADA',
 'Cape Verde',
 'Cayman Islands',
 'Central African Republic',
 'CHAD',
 'CHILE',
 'China, Peoples Republic of',
 'Christmas Island',
 'Cocos Islands',
 'COLOMBIA',
 'Comoros',
 'CONGO',
 'Cook Islands',
 'COSTA RICA',
 'Croatia',
 'CUBA',
 'Cyprus',
 'Czechoslovakia',
 'Czech Republic',
 'Dem Rep of The Congo',
 'DENMARK',
 'Djibouti',
 'Dominica',
 'DOMINICAN REPUBLIC',
 'East Timor',
 'ECUADOR',
 'EGYPT',
 'EL SALVADOR',
 'Equatorial Guinea',
 'ERITREA',
 'ESTONIA',
 'ETHIOPIA',
 'Falkland Islands',
 'FIJI',
 'Finland',
 'FRANCE',
 'French Guiana',
 'FRENCH POLYNESIA',
 'French Southern and Antarctic',
 'Gabon',
 'Gambia',
 'GEORGIA',
 'GERMANY',
 'GHANA',
 'Gibraltar',
 'GREECE',
 'Greenland',
 'Grenada',
 'Guadeloupe',
 'GUAM',
 'GUATEMALA',
 'GUINEA',
 'Guinea-Bissau',
 'GUYANA',
 'HAITI',
 'Holy See',
 'HONDURAS',
 'HONG KONG',
 'HUNGARY',
 'Iceland',
 'INDIA',
 'INDONESIA',
 'IRAN',
 'IRAQ',
 'Ireland',
 'ISRAEL',
 'Italy',
 'Ivory Coast',
 'JAMAICA',
 'JAPAN',
 'JORDAN',
 'KAZAKHSTAN',
 'KENYA',
 'Kiribati',
 'Korea',
 'Kosovo',
 'Kuwait',
 'Kyrgyzstan',
 'Laos',
 'Latvia',
 'LEBANON',
 'Lesotho',
 'Liberia',
 'Libya',
 'Liechtenstein',
 'Lithuania',
 'Luxembourg',
 'Macau',
 'Macedonia',
 'Madagascar',
 'MALAWI',
 'MALAYSIA',
 'Maldives',
 'Mali',
 'MALTA',
 'Mariana Islands, Northern',
 'Marshall Islands',
 'Martinique',
 'MAURITANIA',
 'Mauritius',
 'MEXICO',
 'Micronesia, Federated States of',
 'Moldova',
 'Monaco',
 'MONGOLIA',
 'Montserrat',
 'MOROCCO',
 'Mozambique',
 'Namibia',
 'Nauru',
 'NEPAL',
 'Netherlands',
 'Netherlands Antilles',
 'New Caledonia',
 'NEW ZEALAND',
 'NICARAGUA',
 'Niger',
 'NIGERIA',
 'Niue',
 'North Korea',
 'Norway',
 'Oceania',
 'Oman',
 'PAKISTAN',
 'Palau',
 'PANAMA',
 'Papua New Guinea',
 'Paraguay',
 'PERU',
 'PHILIPPINES',
 'Pitcairn Islands',
 'POLAND',
 'PORTUGAL',
 'Qatar',
 'Reunion',
 'ROMANIA',
 'RUSSIA',
 'Rwanda',
 'SAMOA',
 'San Marino',
 'Sao Tome and Principe',
 'SAUDI ARABIA',
 'Senegal',
 'Serbia and Montenegro',
 'Seychelles',
 'SIERRA LEONE',
 'Singapore',
 'Slovakia',
 'Slovenia',
 'Solomon Islands',
 'SOMALIA',
 'SOUTH AFRICA',
 'South Korea',
 'South Sudan',
 'SPAIN',
 'Sri Lanka',
 'St. Helena',
 'St. Kitts-Nevis',
 'St. Lucia',
 'St. Pierre and Miquelon',
 'St. Vincent-Grenadines',
 'STATELESS',
 'SUDAN',
 'Suriname',
 'Swaziland',
 'Sweden',
 'SWITZERLAND',
 'SYRIA',
 'Taiwan',
 'TAJIKISTAN',
 'Tanzania',
 'THAILAND',
 'TOGO',
 'TONGA',
 'Trinidad and Tobago',
 'TUNISIA',
 'TURKEY',
 'Turkmenistan',
 'Turks and Caicos Islands',
 'Tuvalu',
 'UGANDA',
 'UKRAINE',
 'UNITED ARAB EMIRATES',
 'UNITED KINGDOM',
 'Unknown',
 'Uruguay',
 'Ussr',
 'UZBEKISTAN',
 'Vanuatu',
 'VENEZUELA',
 'VIETNAM',
 'Wallis and Futuna Islands',
 'Western Sahara',
 'Western Samoa',
 'Yemen',
 'Yugoslavia',
 'Zaire',
 'Zambia',
 'ZIMBABWE']

lang = ['Acateco',
 'Achi',
 'Albanian',
 'American Sign Language',
 'Amharic',
 'Amuzgos',
 'Arabic',
 'Armenian',
 'Assyrian',
 'Bengali',
 'Burmese',
 'Cambodian',
 'Cantonese',
 'Castellano',
 'Chatina',
 'Chinese',
 'Creole',
 'Croatian',
 'Czech',
 'Decline to State',
 'English',
 'Faroese',
 'Farsi',
 'French',
 'Garifuna',
 'German',
 'Gujarati',
 'Haitian Creole',
 'Hausa',
 'Hebrew',
 'Hindi',
 'Hmong',
 'Hungarian',
 'Ilocan',
 'Italian',
 'Japanese',
 'Kackchikquel',
 'Khmu',
 'Korean',
 'Lao',
 'Lingala',
 'Macedonian',
 'Mam',
 'Mandarin',
 'Mien',
 'Miskito',
 'Mixtec',
 'Mixteco',
 'Mongolian',
 'Nepali',
 'Other',
 'Other Chinese Languages',
 'Other Non-English',
 'Other Sign Language',
 'Pashto',
 'Pashtu',
 'Polish',
 'Portuguese',
 'Pulaar',
 'Punjabi',
 "Q'anjob'al",
 'Qanjobal',
 'Qeqchi',
 'Quiche',
 'Quiché (Kʼicheʼ)',
 'Qʼeqchiʼ(Kekchi)',
 'Romanian',
 'Russian',
 'Samoan',
 'Somali',
 'Spanish',
 'Swahili',
 'Tagalog',
 'Tajik',
 'Tamil',
 'Thai',
 'Tigrigna',
 'Tigrinya',
 'Triqui',
 'Turkic',
 'Turkish',
 'Twi',
 'Tzeltal',
 'Tzotzil',
 'Ukrainian',
 'Urdu',
 'Uzbek',
 'Vietnamese',
 'Yoruba',
 'Zapoteco']

states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    # Districts and territories
    'District of Columbia': 'DC',
    'Puerto Rico': 'PR',
    'Guam': 'GU',
    'American Samoa': 'AS',
    'U.S. Virgin Islands': 'VI',
    'Northern Mariana Islands': 'MP'
}

cob = ['Afghanistan',
 'Albania',
 'Algeria',
 'Andorra',
 'Angola',
 'Antigua and Barbuda',
 'Argentina',
 'Armenia',
 'Australia',
 'Austria',
 'Azerbaijan',
 'Bahamas',
 'Bahrain',
 'Bangladesh',
 'Barbados',
 'Belarus',
 'Belgium',
 'Belize',
 'Benin',
 'Bhutan',
 'Bolivia',
 'Bosnia and Herzegovina',
 'Botswana',
 'Brazil',
 'Brunei',
 'Bulgaria',
 'Burkina Faso',
 'Burundi',
 "Côte d'Ivoire",
 'Cabo Verde',
 'Cambodia',
 'Cameroon',
 'Canada',
 'Central African Republic',
 'Chad',
 'Chile',
 'China, Peoples Republic of',
 'Colombia',
 'Comoros',
 'Congo (Congo-Brazzaville)',
 'Costa Rica',
 'Croatia',
 'Cuba',
 'Cyprus',
 'Czechia (Czech Republic)',
 'Democratic Republic of the Congo',
 'Denmark',
 'Djibouti',
 'Dominica',
 'Dominican Republic',
 'Ecuador',
 'Egypt',
 'El Salvador',
 'Equatorial Guinea',
 'Eritrea',
 'Estonia',
 'Eswatini (fmr. "Swaziland")',
 'Ethiopia',
 'Fiji',
 'Finland',
 'France',
 'Gabon',
 'Gambia',
 'Georgia',
 'Germany',
 'Ghana',
 'Greece',
 'Grenada',
 'Guatemala',
 'Guinea',
 'Guinea-Bissau',
 'Guyana',
 'Haiti',
 'Holy See',
 'Honduras',
 'Hungary',
 'Iceland',
 'India',
 'Indonesia',
 'Iran',
 'Iraq',
 'Ireland',
 'Israel',
 'Italy',
 'Jamaica',
 'Japan',
 'Jordan',
 'Kazakhstan',
 'Kenya',
 'Kiribati',
 'Kuwait',
 'Kyrgyzstan',
 'Laos',
 'Latvia',
 'Lebanon',
 'Lesotho',
 'Liberia',
 'Libya',
 'Liechtenstein',
 'Lithuania',
 'Luxembourg',
 'Madagascar',
 'Malawi',
 'Malaysia',
 'Maldives',
 'Mali',
 'Malta',
 'Marshall Islands',
 'Mauritania',
 'Mauritius',
 'Mexico',
 'Micronesia',
 'Moldova',
 'Monaco',
 'Mongolia',
 'Montenegro',
 'Morocco',
 'Mozambique',
 'Myanmar (formerly Burma)',
 'Namibia',
 'Nauru',
 'Nepal',
 'Netherlands',
 'New Zealand',
 'Nicaragua',
 'Niger',
 'Nigeria',
 'North Korea',
 'North Macedonia',
 'Norway',
 'Oman',
 'Pakistan',
 'Palau',
 'Palestine State',
 'Panama',
 'Papua New Guinea',
 'Paraguay',
 'Peru',
 'Philippines',
 'Poland',
 'Portugal',
 'Qatar',
 'Romania',
 'Russia',
 'Rwanda',
 'Saint Kitts and Nevis',
 'Saint Lucia',
 'Saint Vincent and the Grenadines',
 'Samoa',
 'San Marino',
 'Sao Tome and Principe',
 'Saudi Arabia',
 'Senegal',
 'Serbia',
 'Seychelles',
 'Sierra Leone',
 'Singapore',
 'Slovakia',
 'Slovenia',
 'Solomon Islands',
 'Somalia',
 'South Africa',
 'South Korea',
 'South Sudan',
 'Spain',
 'Sri Lanka',
 'Sudan',
 'Suriname',
 'Sweden',
 'Switzerland',
 'Syria',
 'Tajikistan',
 'Tanzania',
 'Thailand',
 'Timor-Leste',
 'Togo',
 'Tonga',
 'Trinidad and Tobago',
 'Tunisia',
 'Turkey',
 'Turkmenistan',
 'Tuvalu',
 'Uganda',
 'Ukraine',
 'United Arab Emirates',
 'United Kingdom',
 'United States of America',
 'Uruguay',
 'Uzbekistan',
 'Vanuatu',
 'Venezuela',
 'Vietnam',
 'Yemen',
 'Zambia',
 'Zimbabwe',
 'none',
 'Other']

cf['Country of Birth'].fillna('--None--', inplace=True)
cf['Country of Birth'] = cf['Country of Birth'].astype(str)

lgbt = {'No':'Heterosexual',' ':'Heterosexual','Yes':'LGBTQ+'}

# your list of correctly spelled countries
correct_countries = cob

def correct_spelling_partial(name):
    if name == 'nan':  # return NaN for 'nan' values
        return np.nan
    elif 'Mexican' in name:  # special handling for 'Mexican'
        return 'Mexico'
    else:
        # sort correct_countries by length (longest first)
        correct_countries_sorted = sorted(correct_countries, key=len, reverse=True)
        for country in correct_countries_sorted:
            if country in name:
                return country
        return name  # if no partial match found, return the original name

cf['Country of Birth Corrected'] = cf['Country of Birth'].apply(correct_spelling_partial)

def correct_spelling_case_sensitive(name):
    if name is np.nan:
        return np.nan
    elif 'China' in name:
        return 'China, Peoples Republic of'
    elif 'Czechia (Czech Republic)' in name:
        return 'CZECHOSLOVAKIA'
    elif 'Democratic Republic of the ' in name:
        return 'Dem Rep of The Congo' 
    elif 'Congo (Congo-Brazzaville)' in name:
        return 'CONGO'
    elif "Côte d'Ivoire" in name:
        return 'Ivory Coast'
    else:
        max_ratio = 0
        closest_country = ""
        for correct_name in ice_cob:
            ratio = SequenceMatcher(None, name.lower(), correct_name.lower()).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
                closest_country = correct_name
        return closest_country if max_ratio > 0.8 else name

cf['ICE Country of Birth Corrected'] = cf['Country of Birth Corrected'].apply(correct_spelling_case_sensitive)


def correct_language_spelling(name):
    if name is np.nan:  # return NaN for 'nan' values
        return np.nan
    else:
        max_ratio = 0
        closest_language = ""
        for correct_name in lang:
            ratio = SequenceMatcher(None, name.lower(), correct_name.lower()).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
                closest_language = correct_name
        return closest_language if max_ratio > 0.8 else name

cf['Language Corrected'] = cf['Language'].apply(correct_language_spelling)

def fix_rac(rac):
    if rac == 'Hispanic':
        return 'Latinx'
    if rac == 'Asian (not Pacific Islander)':
        return 'Asian/Pacific Islander'
    if rac == 'Pacific Islander':
        return 'Asian/Pacific Islander'
    if rac == 'White':
        return 'Other'
    if rac == 'Black (not Afro-Latinx)':
        return 'Black'
    if rac == 'Afro-Latinx':
        return 'Black'
    if rac == ' ':
        return np.nan
    if rac == 'Multi-racial':
        return 'Other'
    if rac == 'Mid. Eastern / N. African (MENA)':
        return 'Other'
    else:
        return rac
cf['Ethnicity Corrected'] = cf['Ethnicity'].apply(fix_rac)

def fix_mar(sta):
    if sta is np.nan:
        return '--None--'
    elif sta == 'None':
        return '--None--'
    elif sta == ' ':
        return '--None--'
    elif sta == 'Legally Married':
        return 'Married'
    elif sta == 'Other (Specify Below)':
        return '--None--'
    elif sta == 'Divorced':
        return '--None--'
    elif sta == 'Widowed':
        return '--None--'
    elif sta == 'Partner':
        return 'Partnered'
    else:
        return sta
cf['Marital Status Corrected'] = cf['Marital Status'].apply(fix_mar)

cf['Mailing State/Province Corrected'] = cf['Mailing State/Province'].replace(states).str.upper()

cf['Gender Identity/Expression'].replace(' ', np.nan,inplace=True)

cf['Sexual Orientation Corrected'] = cf['Sexual Orientation'].replace(lgbt)

status = {'Undocumented':'No Status','LPR':'Permanent Resident','Unsure':'Unknown', 'Other (Specify below)':'Other',' ':np.nan, 'Visa Overstay':'Other'}

cf['Immigration Status Corrected'] = cf['Immigration Status'].replace(status)

cf['ICE Find'] = 'TRUE'

cf['Presumed Detained'] = 'TRUE'

final = cf[['Client Since', 'First Name', 'Last Name', 'ICE Cohort', 'Client Source', 'Language Corrected','Ethnicity Corrected','Birthdate','Mailing City',
    'Mailing County', 'Mailing State/Province Corrected','Mailing Zip/Postal Code','Country of Birth Corrected','ICE Country of Birth Corrected','Presumed Detained','ICE Find',
    'Immigration Status Corrected','Detention Date','Detention Statute','Immigration Arrest Circumstances','Marital Status Corrected','Gender Identity/Expression',
    'Sexual Orientation Corrected','Emergency Contact Name','Emergency Contact Phone']]

final.columns = final.columns.str.replace(' Corrected', '')

mp_col = {'A Number':'Alien_Number__c','First Name':'FirstName', 'Last Name':'LastName','ICE Cohort':'ICE_Cohort__c','Client Source':'Client_Source__c','Language':'Language__c','Ethnicity':'Ethnicity__c','Mailing City':'MailingCity','Mailing County':'Mailing_County__c','Mailing State/Province':'MailingState','Mailing Zip/Postal Code':'MailingPostalCode','Country of Birth':'mp_Country_of_Birth__c','ICE Country of Birth':'Country_of_Birth__c','ICE Find':'ICE_scrape__c','Immigration Status':'Immigration_Status__c','Detention Date':'Detention_Date__c','Immigration Arrest Circumstances':'Immigration_Arrest_Circumstances__c','Marital Status':'Marital_Status__c','Gender Identity/Expression':'Gender__c','Sexual Orientation':'sexual_orientation__c','Emergency Contact Name':'AssistantName','Emergency Contact Phone':'AssistantPhone'}

final = final.rename(columns=mp_col)

final = final.reset_index().dropna(subset=['A Number']).set_index('A Number')

final.to_csv('/Users/stjames/Dropbox/Pablo/data/legal-clinic-processing/final.csv')


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

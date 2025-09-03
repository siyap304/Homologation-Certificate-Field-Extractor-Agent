import os
import base64
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from openai import AzureOpenAI
from io import BytesIO
import json
import pandas as pd
from datetime import datetime, date
import pycountry
from pydantic import BaseModel, field_validator, model_validator
from typing import Union
from dateutil.relativedelta import relativedelta
from PIL import Image

class Certificate_Validate(BaseModel):
    issue_date: Union[date, str]
    expiry_date: Union[date, str]
    country: str
    model_number: str
    validity: Union[str, int]

    @field_validator('issue_date', 'expiry_date', mode="before")
    def validate_dates(cls, value):
        if value == "None":
            return value
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError('Dates must be in YYYY-MM-DD format')
        
    @model_validator(mode="after")
    def check_expiry_after_issue(cls, values):
        issue = values.issue_date
        expiry = values.expiry_date

        if isinstance(issue, date) and isinstance(expiry, date):
            if expiry <= issue:
                raise ValueError("expiry_date must be after issue_date")
        
        return values

    @field_validator("country", mode="before")
    def validate_country(cls, value):
        if not isinstance(value, str):
            raise ValueError("Country must be a string")

        countries = [c.strip() for c in value.split(",") if c.strip()]
        invalid = []

        for country in countries:
            try:
                if country == "Palestine" or "Palestine, State of":
                    continue
                pycountry.countries.lookup(country)
            except LookupError:
                invalid.append(country)

        if invalid:
            raise ValueError(f"Invalid country/countries: {', '.join(invalid)}")

        return value 

class DataExtractor:
    def __init__(self):
        load_dotenv()

        self.OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        self.AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
        self.AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
        self.client = AzureOpenAI(
            api_key=self.OPENAI_API_KEY,
            api_version=self.OPENAI_API_VERSION,
            azure_endpoint=self.AZURE_OPENAI_API_BASE,
            azure_deployment=self.AZURE_DEPLOYMENT
        )

    def base64_images_roundtrip(self, base64_strings):
        """
        Takes a list of base64-encoded image strings,
        decodes each to a PIL Image,
        then re-encodes each as a base64 JPEG string.
        Returns the list of base64 JPEG strings.
        """
        output_base64_images = []
        for b64_str in base64_strings:
            if not isinstance(b64_str, str):
                raise ValueError(f"Expected string but got {type(b64_str)}")

            # Decode base64 string to PIL Image
            img_data = base64.b64decode(b64_str)
            img = Image.open(BytesIO(img_data))

            # Re-encode PIL Image as base64 JPEG
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            encoded_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            output_base64_images.append(encoded_str)

        return output_base64_images


    def prepare_messages(self, base64_images):
        messages = [
            {
                "role": "system",
                "content": """You are an intelligent image data extraction system. Your task is to extract text fields from the images of Homolagation Certificates and convert the data into JSON format. The JSON should contain the following fields:
                            country_name (the country to which this certificate belongs to, but not the supplier country/producer country/fabricated country)
                            certificate_issue_date (last_approval_date or Issuance_date or renewal date)
                            certificate_expiry_date
                            certification_number
                            supplier_name (or manufacturer_name or producer_name)
                            product_name
                            Model_number (or Model_name or Model or model information)
                            brand_name
                            certificate_holder_name
                            certificate_validity_in_years
                            validity_in_months
                            If no information is available for a field, return a string "None".
                            If there are multiple answers for a field, return a string with comma seperated values.

                            Special Instructions:

                            If the certificate expiry date is not directly available, look for text indicating the validity period (e.g., "certificate is valid for 2 years" or "certificate is valid for 5 years"). Calculate the expiry date based on the certificate_issue_date.
                            Be extra attentive while extracting numerical values, Please Double check all fields as they are crucial for the JSON output . 
                            Numerical values like 5,6 or 8 might seem confusing due to low resolution, please be mindful of it and double check. 
                            Some dates / fields might be handwritten. Please be mindful of it and double check the value if it is handwritten.
                            In some certificates, the fond used is stylized, partially open circle glyph. So, in a tightly cropped line, alphabets might appear to have similar shapes. Please be mindful of this and dont merge/misread/skip any alphabets due to this. If such a case appears, you can extract the field again to check.
                            The certificates may have decorative border and background patterns introduce noise. Please be mindful of this and ensure that this does not cause an issue in your judgement.
                            The json file you return should be made of two more jsons. The first json will be 'required' and the second json will be 'extracted'.
                            Both the jsons will have the same fields listed above. The values in the extracted json should be the actual text from the certificate. 
                            In the extracted json, please dont add any text on your own. It should be exactly as visible in the certificate. If the text is not in english, please DO NOT transalate it for the extracted json. 
                            The required json will be the same fields extracted from the certificate but in English. For the required json, please return the country name according to those listed in ISO 3166 standard and the date in YYYY-MM-DD format.
                            
                            Example output json:
                            {
                                extracted:{
                                    country_name:
                                    certificate_issue_date: 
                                    certificate_expiry_date: (as seen in the certificate, DO NOT change/ remove any symbols. DO NOT calculate, return "None" if not present)
                                    certificate_validity_in_years: (give the main key words. DO NOT calculate, return "None" if not present)
                                    validity_in_months: 
                                    and so on
                                },
                                required:{
                                    country_name: (ISO 3166 standard country name, DO NOT GIVE THE CODE)
                                    certificate_issue_date: (YYYY-MM-DD format)
                                    certificate_expiry_date: (YYYY-MM-DD format)
                                    certificate_validity_in_years: 
                                    validity_in_months: (please calculate if not given)
                                    and so on
                                }
                            }
                            """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract relevant info from these images."},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in base64_images]
                ]
            }
        ]
        return messages

    def extract_data(self, images):
        try:
            base64_images = self.base64_images_roundtrip(images)
            messages = self.prepare_messages(base64_images)
            response = self.client.chat.completions.create(
                model=self.AZURE_DEPLOYMENT,
                messages=messages,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            data = json.loads(content)
            required_data = data.get("required", {})
            df = pd.DataFrame([required_data])

            expiry_date_df = df["certificate_expiry_date"].iloc[0]
            issue_date_df = df["certificate_issue_date"].iloc[0]
            country_name = df["country_name"].iloc[0]
            model_no = df["Model_number"].iloc[0]
            validity = df["validity_in_months"].iloc[0]
            if validity in [None, "None"]:
                if expiry_date_df in [None, "None"] or issue_date_df in [None, "None"]:
                    validity = "None"
                else:
                    expiry_date = pd.to_datetime(expiry_date_df)
                    issue_date = pd.to_datetime(issue_date_df)
                    
                    delta = relativedelta(expiry_date, issue_date)
                    validity = delta.years * 12 + delta.months
            
            Certificate_Validate(issue_date = issue_date_df, expiry_date = expiry_date_df, country = country_name, model_number = model_no, validity=validity)
            
            return content

        except Exception as e:
            print(e)
            error_response = {
                "required": {
                "error": "An error occurred while processing the request.",
                "details": str(e)
                },
                "extracted": {
                "error": "An error occurred while processing the request.",
                "details": str(e)
                }
            }
            return json.dumps(error_response)
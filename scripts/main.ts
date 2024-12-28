import {
    Browser,
    Page,
    PuppeteerWebBaseLoader,
  } from "@langchain/community/document_loaders/web/puppeteer";
  import dotenv from "dotenv";
  import { OpenAI } from "openai";
  import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
  import { DataAPIClient } from "@datastax/astra-db-ts";
  
  dotenv.config();
  
  const muscleData = [
    "https://ja.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC%E3%81%AE%E3%83%9C%E3%83%87%E3%82%A3%E3%83%93%E3%83%AB%E3%83%80%E3%83%BC",
    "https://ja.wikipedia.org/wiki/%E6%9C%A8%E6%BE%A4%E5%A4%A7%E7%A5%90"
  ];
  
  const scrapePage = async () => {
    const pageData = [];
    
    for await (const url of muscleData) {
      try {
        const loader = new PuppeteerWebBaseLoader(url, {
          launchOptions: {
            headless: true,
            args: ["--no-sandbox", "--disable-setuid-sandbox"],
          },
          gotoOptions: {
            waitUntil: "domcontentloaded",
          },
          evaluate: async (page: Page, browser: Browser) => {
            const result = await page.evaluate(() => document.body.innerHTML);
            await browser.close();
            return result;
          },
        });
    
        const data = await loader.scrape();
        pageData.push(data);
      } catch (error) {
        console.error(`Error scraping ${url}:`, error);
      }
    }
    
    return pageData;
  };
  
  const {
    ASTRA_DB_NAMESPACE,
    ASTRA_DB_COLLECTION,
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_APPLICATION_TOKEN,
    OPENAI_API_KEY,
  } = process.env;
  
  const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
  });
  
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512,
    chunkOverlap: 100,
  });
  
  const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
  const db = client.db(ASTRA_DB_API_ENDPOINT!, { namespace: ASTRA_DB_NAMESPACE });
  
  const converVectorAndSave = async (pageData: string[]) => {
    for (const page of pageData) {
      const pageChunks = await splitter.splitText(page);
      const collection = await db.collection(ASTRA_DB_COLLECTION!);
  
      for await (const chunk of pageChunks) {
        try {
        const embedding = await openai.embeddings.create({
          model: "text-embedding-3-small",
          input: chunk,
          encoding_format: "float",
        });
  
        const vector = embedding.data[0].embedding;
  
        await collection.insertOne({
          $vector: vector,
          text: chunk,
        });
      }catch (error) {
        console.error("Error inserting data:", error);
      }
    }
  }
};
const createCollection = async () => {
    try {
      const collections = await db.listCollections();
      const collectionExists = collections.some(
        (collection) => collection.name === ASTRA_DB_COLLECTION
      );
  
      if (!collectionExists) {
        const res = await db.createCollection(ASTRA_DB_COLLECTION!, {
          vector: {
            dimension: 1536,
            metric: "cosine",
          },
        });
        console.log("Collection created:", res);
      } else {
        console.log(`Collection '${ASTRA_DB_COLLECTION}' already exists.`);
      }
    } catch (error) {
      console.error("Error creating collection:", error);
    }
  };
  
  const main = async () => {
    try{
    await createCollection();
    const pageData = await scrapePage();
    await converVectorAndSave(pageData);
  }catch (error) {
    console.error("Error in main function:", error);
  }
};
  
  main();
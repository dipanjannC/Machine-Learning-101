package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
)

// ResultData structure used for result text format
type ResultData struct {
	LuceneID                      string  `json:"Lucene ID"`
	BERTSQuADAnswerWithHighlights string  `json:"BERT-SQuAD Answer with Highlights"`
	Confidence                    float64 `json:"Confidence"`
	Title                         string  `json:"Title/Link"`
	Abstract                      string  `json:"Abstract"`
}

// Data structure used for richText()
type Data struct {
	RichText string `json:"rich_text"`
	Warning  string `json:"warning"`
	Result   string `json:"result"`
}

// Info structure used for detailedText()
type Info struct {
	ID    string `json:"id"`
	Title string `json:"title"`
	Text  string `json:"text"`
}

// GlobalQuestion variable used for storing the question asked
var GlobalQuestion string

// InitApp function used for initializing the GlobalQuestion variable
func InitApp(val string) string {
	GlobalQuestion = val
	return GlobalQuestion
}

func richText(responseData []byte) interface{} {
	var dataObject Data
	json.Unmarshal(responseData, &dataObject)

	resultTexts := dataObject.Result
	var someStr []ResultData
	_ = json.Unmarshal([]byte(resultTexts), &someStr)

	fmt.Println(dataObject.RichText)
	fmt.Println(len(resultTexts))

	return someStr
}

func richTextHandler(c *gin.Context) {
	c.Request.ParseForm()
	question := c.Request.Form.Get("question")
	InitApp(question)
	jsonRequest, err := json.Marshal(map[string]string{
		"question": question,
	})
	if err != nil {
		fmt.Print(err.Error())
		os.Exit(1)
	}

	fmt.Println(string(jsonRequest))
	if len(question) <= 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "question field is blank"})
	}

	response, err := http.Post("http://app:5000/rich-text", "application/json; charset=utf-8", bytes.NewBuffer(jsonRequest))
	if err != nil {
		fmt.Print(err.Error())
		os.Exit(1)
	}

	responseData, err := ioutil.ReadAll(response.Body)
	if err != nil {
		log.Fatal(err)
	}

	c.HTML(http.StatusOK, "view_articles.html", gin.H{
		"ResultData": richText(responseData),
		"Question":   question,
	})
}

func detailedText(responseData []byte) interface{} {
	var infoObject Info
	json.Unmarshal(responseData, &infoObject)

	fmt.Println(string(responseData))
	fmt.Println("Lucene ID: ", infoObject.ID)
	fmt.Println("Paper title: ", infoObject.Title)

	return infoObject
}

func detailedTextHandler(c *gin.Context) {
	id := c.Params.ByName("id")
	question := GlobalQuestion
	jsonRequest, err := json.Marshal(map[string]string{
		"question": question,
		"doc_id":   id,
	})
	fmt.Println(string(jsonRequest))

	response, err := http.Post("http://app:5000/detailed-text", "application/json; charset=utf-8", bytes.NewBuffer(jsonRequest))
	if err != nil {
		fmt.Print(err.Error())
		os.Exit(1)
	}

	responseData, err := ioutil.ReadAll(response.Body)
	if err != nil {
		log.Fatal(err)
	}

	res := strings.Contains(string(responseData), "Internal Server Error")

	if len(responseData) > 0 && res == false {
		c.HTML(http.StatusOK, "view_article.html", gin.H{
			"Info": detailedText(responseData),
		})
	} else {
		c.JSON(http.StatusNotFound, gin.H{"error": "This article doesn't exists"})
	}
}

func index(c *gin.Context) {
	c.HTML(http.StatusOK, "index.html", nil)
}

func main() {
	router := gin.Default()
	router.LoadHTMLGlob("views/*")
	router.GET("/", index)
	router.POST("/search", richTextHandler)
	router.GET("/search/:id", detailedTextHandler)
	router.Run(":3000")
}
